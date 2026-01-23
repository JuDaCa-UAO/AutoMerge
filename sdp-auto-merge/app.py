from __future__ import annotations

import math
import re
from typing import Any, Dict, List

import streamlit as st

from servicedesk_merge.config import load_settings, SettingsError
from servicedesk_merge.sdp_client import ServiceDeskClient, ServiceDeskApiError
from servicedesk_merge.ai_gpt import refine_candidate_groups_with_ai, AIError
from servicedesk_merge.heuristic_helper import (
    build_candidate_groups,
    normalize_subject,
    requester_is_servicedesk,
    status_is_open,
    technician_is_unassigned,
)

IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


# ----------------------------- Helpers -----------------------------
def safe_get(d: dict, *path: str, default=""):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def pick_name(obj, default=""):
    if isinstance(obj, dict):
        return obj.get("name") or obj.get("display_value") or default
    if obj is None:
        return default
    return str(obj)


def created_display(t: dict) -> str:
    ct = safe_get(t, "created_time", default=None)
    if isinstance(ct, dict):
        return str(ct.get("display_value") or ct.get("value") or "")
    return str(ct or "")


def summarize_ticket(t: dict) -> dict:
    site = safe_get(t, "site", "name", default="") or pick_name(safe_get(t, "site", default=""), "")
    status = pick_name(safe_get(t, "status", default=""), "")
    requester = pick_name(safe_get(t, "requester", default=""), "")
    technician = pick_name(safe_get(t, "technician", default=""), "")

    return {
        "id": str(safe_get(t, "id", default="")).strip(),
        "subject": str(safe_get(t, "subject", default="")).strip(),
        "site": str(site).strip(),
        "status": str(status).strip(),
        "requester": str(requester).strip(),
        "technician": str(technician).strip(),
        "created": created_display(t).strip(),
    }


def fetch_pool(client: ServiceDeskClient, pool_size: int, page_size: int, start_index: int) -> list[dict]:
    fetched = 0
    cursor = int(start_index)
    est_pages = int(math.ceil(int(pool_size) / int(page_size)))
    prog = st.progress(0, text="Iniciando...")

    out: list[dict] = []
    for p in range(est_pages):
        batch = min(int(page_size), int(pool_size) - fetched)
        prog.progress(min(1.0, (p / max(est_pages, 1))), text=f"Cargando página {p+1}/{est_pages}...")

        items, more, _payload = client.list_requests_page(start_index=cursor, row_count=batch)
        if not items:
            break

        out.extend([summarize_ticket(x) for x in items])
        fetched += len(items)

        if fetched >= int(pool_size) or not more:
            break
        cursor += batch

    prog.progress(1.0, text=f"Listo. Tickets cargados: {len(out)}")
    return out


def derive_sites_from_tickets(tickets: list[dict]) -> list[str]:
    return sorted({t.get("site", "").strip() for t in tickets if t.get("site", "").strip()})


def _to_int_id(x: str) -> int:
    try:
        return int(str(x).strip())
    except Exception:
        return 10**18


def remove_ids_from_list(tickets: list[dict], ids_to_remove: set[str]) -> list[dict]:
    return [t for t in tickets if str(t.get("id", "")).strip() not in ids_to_remove]


def replace_or_insert_ticket(tickets: list[dict], updated: dict) -> list[dict]:
    uid = str(updated.get("id", "")).strip()
    if not uid:
        return tickets
    out = []
    replaced = False
    for t in tickets:
        if str(t.get("id", "")).strip() == uid:
            out.append(updated)
            replaced = True
        else:
            out.append(t)
    if not replaced:
        out.insert(0, updated)
    return out


def refresh_parent_from_api(client: ServiceDeskClient, parent_id: str) -> dict | None:
    try:
        req = client.get_request(parent_id)
        return summarize_ticket(req)
    except Exception:
        return None


def apply_merge_local_update(client: ServiceDeskClient, site_name: str, parent_id: str, child_ids: list[str]) -> None:
    parent_id = str(parent_id).strip()
    child_set = {str(x).strip() for x in child_ids if str(x).strip()}
    child_set.discard(parent_id)

    pool = st.session_state.get("pool_data", []) or []
    src_pool = st.session_state.get("site_source_pool", []) or pool

    # site cache (por sitio)
    site_cache: Dict[str, List[dict]] = st.session_state.get("site_cache", {}) or {}
    site_list = site_cache.get(site_name, []) or []

    pool = remove_ids_from_list(pool, child_set)
    src_pool = remove_ids_from_list(src_pool, child_set)
    site_list = remove_ids_from_list(site_list, child_set)

    updated_parent = refresh_parent_from_api(client, parent_id)
    if updated_parent:
        pool = replace_or_insert_ticket(pool, updated_parent)
        src_pool = replace_or_insert_ticket(src_pool, updated_parent)
        if updated_parent.get("site") == site_name:
            site_list = replace_or_insert_ticket(site_list, updated_parent)

    st.session_state["pool_data"] = pool
    st.session_state["site_source_pool"] = src_pool
    site_cache[site_name] = site_list
    st.session_state["site_cache"] = site_cache


def update_groups_after_merge(site_name: str, parent_id: str, child_ids: list[str]) -> None:
    """
    Remapea grupos IA localmente sin volver a llamar IA.
    """
    merged_children = {str(x).strip() for x in child_ids if str(x).strip()}
    merged_children.discard(str(parent_id).strip())

    all_groups_by_site: Dict[str, List[dict]] = st.session_state.get("ai_groups_by_site", {}) or {}
    groups = all_groups_by_site.get(site_name, []) or []

    new_groups = []
    for g in groups:
        if not isinstance(g, dict):
            continue
        g_parent = str(g.get("parent_id", "") or "").strip()
        g_children = [str(x).strip() for x in (g.get("child_ids") or []) if str(x).strip()]
        g_children = [c for c in g_children if c not in merged_children and c != g_parent]

        # si el padre fue mergeado como hijo en otra operación, marcar para revisión (no borramos)
        if g_parent in merged_children:
            g2 = dict(g)
            g2["child_ids"] = g_children
            g2["reason"] = (g2.get("reason", "") or "") + " | PARENT_WAS_MERGED_REVIEW"
            new_groups.append(g2)
            continue

        g2 = dict(g)
        g2["child_ids"] = g_children
        new_groups.append(g2)

    all_groups_by_site[site_name] = new_groups
    st.session_state["ai_groups_by_site"] = all_groups_by_site


def eligible_for_ai(t: dict) -> bool:
    return (
        status_is_open(t.get("status", ""))
        and technician_is_unassigned(t.get("technician", ""))
        and requester_is_servicedesk(t.get("requester", ""))
    )


def compact_ticket_for_ai(t: dict) -> dict:
    return {
        "id": str(t.get("id", "")).strip(),
        "subject": str(t.get("subject", "")).strip(),
        "created": str(t.get("created", "")).strip(),
        "status": str(t.get("status", "")).strip(),
        "requester": str(t.get("requester", "")).strip(),
        "technician": str(t.get("technician", "")).strip(),
    }


# ----------------------------- Manual board UI -----------------------------
def render_manual_multimerge_board(
    *,
    client: ServiceDeskClient,
    selected_site: str,
    site_tickets: list[dict],
) -> None:
    """
    Multi-merge manual:
    - por defecto te muestra solo Open+Unassigned
    - puedes incluir asignados si lo necesitas (checkbox)
    """
    st.markdown("### 📌 Compilado (todos los tickets del sitio) + Multi-merge manual")

    include_assigned = st.checkbox(
        "Incluir tickets con technician asignado (solo para merge manual)",
        value=False,
        key=f"include_assigned__{selected_site}",
        help="La IA seguirá siendo estricta, pero tú puedes mergear manualmente si lo necesitas.",
    )

    # Scope para merge manual (para que sea práctico)
    if include_assigned:
        manual_scope = [t for t in site_tickets if (t.get("status", "").strip().lower() == "open")]
    else:
        manual_scope = [t for t in site_tickets if eligible_for_ai(t)]

    excluded_count = len(site_tickets) - len(manual_scope)
    st.caption(f"Tickets disponibles en tablero manual: {len(manual_scope)} (excluidos por filtros: {excluded_count})")

    compiled = sorted(manual_scope, key=lambda t: _to_int_id(t.get("id", "")))
    st.dataframe(compiled, use_container_width=True, height=300)

    all_ids = [str(t.get("id")) for t in compiled if str(t.get("id", "")).strip()]
    if len(all_ids) < 2:
        st.info("No hay suficientes tickets en el scope actual para hacer merges manuales.")
        return

    # Plan por sitio
    plan_by_site: Dict[str, Dict[str, List[str]]] = st.session_state.get("manual_plan_by_site", {}) or {}
    plan = plan_by_site.get(selected_site, {}) or {}

    parents_key = f"manual_parents__{selected_site}"

    parents = st.multiselect(
        "Selecciona uno o varios PADRES",
        options=all_ids,
        default=[p for p in plan.keys() if p in all_ids],
        key=parents_key,
    )

    # limpiar plan si quitaron padres (NO tocar session_state del widget, solo el plan)
    plan = {p: kids for p, kids in plan.items() if p in parents}

    # evitar hijos repetidos en distintos padres
    used_children = set()
    for p, kids in plan.items():
        for k in kids:
            used_children.add(k)

    st.markdown("#### Asignación de hijos por cada padre")
    for p in parents:
        other_parents = set(parents) - {p}
        current = [x for x in (plan.get(p, []) or []) if x in all_ids]

        available = [
            x for x in all_ids
            if x != p and x not in other_parents and x not in (used_children - set(current))
        ]

        # limpiar current con available
        current = [x for x in current if x in available]

        kids = st.multiselect(
            f"Hijos para el padre {p}",
            options=available,
            default=current,
            key=f"manual_children__{selected_site}__{p}",
        )

        # actualizar used_children
        for x in plan.get(p, []) or []:
            used_children.discard(x)
        for x in kids:
            used_children.add(x)

        plan[p] = kids

    plan_by_site[selected_site] = plan
    st.session_state["manual_plan_by_site"] = plan_by_site

    st.markdown("#### Resumen del plan")
    summary = [{"parent": p, "children": ", ".join(kids), "children_count": len(kids)} for p, kids in plan.items() if kids]
    if summary:
        st.dataframe(summary, use_container_width=True, height=160)
    else:
        st.info("Aún no has asignado hijos a ningún padre.")

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("🧹 Limpiar plan manual", type="secondary", key=f"clear_plan__{selected_site}"):
            plan_by_site[selected_site] = {}
            st.session_state["manual_plan_by_site"] = plan_by_site
            st.rerun()

    with c2:
        if st.button(
            "🚀 Ejecutar merges del plan manual",
            type="primary",
            key=f"run_plan__{selected_site}",
            disabled=not bool(summary),
        ):
            ok, fail = 0, 0
            # ejecutar en orden por ID (más viejo primero)
            operations = [(p, plan[p]) for p in plan.keys() if plan.get(p)]
            operations.sort(key=lambda x: _to_int_id(x[0]))

            current_ids = {str(t.get("id")) for t in site_tickets}
            for parent_id, child_ids in operations:
                child_ids = [c for c in child_ids if c in current_ids and c != parent_id]
                if not child_ids:
                    continue

                try:
                    _res = client.merge_requests(parent_id=parent_id, child_ids=child_ids)
                    ok += 1

                    apply_merge_local_update(client, selected_site, parent_id, child_ids)
                    update_groups_after_merge(selected_site, parent_id, child_ids)

                    # actualizar universe local (para siguientes operaciones)
                    current_ids -= set(child_ids)

                except ServiceDeskApiError as e:
                    fail += 1
                    st.error(f"Fallo merge padre={parent_id} hijos={child_ids}\n{e}")

            st.success(f"Listo. Merges OK: {ok}, fallidos: {fail}")

            # limpiar plan del sitio (sin tocar widget keys)
            plan_by_site[selected_site] = {}
            st.session_state["manual_plan_by_site"] = plan_by_site
            st.rerun()


# ----------------------------- UI -----------------------------
st.set_page_config(page_title="SDP Auto-Merge (Hybrid)", layout="wide")
st.title("SDP Auto-Merge (Heurístico + IA)")

try:
    settings = load_settings()
    client = ServiceDeskClient(
        base_url=settings.api_base_url,
        auth_token=settings.auth_token,
        portal_id=settings.portal_id,
        verify_ssl=settings.verify_ssl,
        timeout_seconds=settings.timeout_seconds,
    )
except SettingsError as e:
    st.error(f"No pude leer .env: {e}")
    st.info("Crea un .env basado en .env.example y vuelve a ejecutar.")
    st.stop()

tab_pool, tab_site = st.tabs(["📥 Pool general", "🏢 Por sitio (híbrido)"])

# ---------------- TAB 1 ----------------
with tab_pool:
    st.subheader("Pool general de tickets")

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        pool_size = st.number_input("Tamaño del pool (total)", min_value=1, max_value=5000, value=500, step=50)
    with c2:
        page_size = st.number_input("Tamaño por página API (<=100)", min_value=1, max_value=100, value=100, step=10)
    with c3:
        start_index = st.number_input("Start index", min_value=1, value=1, step=1)
    with c4:
        debug = st.checkbox("Debug request", value=False)

    if st.button("Cargar pool", type="primary"):
        try:
            st.session_state["pool_data"] = fetch_pool(client, int(pool_size), int(page_size), int(start_index))
            st.session_state["site_source_pool"] = st.session_state["pool_data"]
            st.session_state["site_cache"] = {}  # cache por sitio
            st.success("Pool cargado.")
        except (ServiceDeskApiError, ValueError) as e:
            st.error(str(e))

    if debug and client.last_http:
        st.caption("Último request (debug)")
        st.code(
            f"METHOD: {client.last_http.method}\nURL: {client.last_http.url}\nPARAMS: {client.last_http.params}",
            language="text",
        )

    data = st.session_state.get("pool_data", [])
    if data:
        st.dataframe(data, use_container_width=True, height=520)

# ---------------- TAB 2 ----------------
with tab_site:
    st.subheader("Por sitio (heurístico + IA) + merge revisado")

    src_pool = st.session_state.get("site_source_pool", [])
    if not src_pool:
        st.warning("Primero carga un pool en el tab 'Pool general'.")
        st.stop()

    sites = derive_sites_from_tickets(src_pool)
    if not sites:
        st.warning("No pude derivar sitios desde el pool (campo 'site' vacío).")
        st.stop()

    selected_site = st.selectbox("Selecciona un sitio", options=sites)

    # cargar tickets del sitio desde cache o desde pool
    site_cache: Dict[str, List[dict]] = st.session_state.get("site_cache", {}) or {}
    if selected_site not in site_cache:
        site_cache[selected_site] = [t for t in src_pool if t.get("site") == selected_site]
        st.session_state["site_cache"] = site_cache

    site_tickets = site_cache.get(selected_site, []) or []
    if not site_tickets:
        st.info("Este sitio no tiene tickets en el pool cargado.")
        st.stop()

    st.markdown("### Tickets del sitio (referencia)")
    st.dataframe(sorted(site_tickets, key=lambda t: _to_int_id(t.get("id", ""))), use_container_width=True, height=280)

    st.divider()

    # 1) SIEMPRE mostrar tablero manual primero
    render_manual_multimerge_board(client=client, selected_site=selected_site, site_tickets=site_tickets)

    st.divider()

    # 2) Híbrido: heurístico + IA (solo sobre elegibles)
    st.markdown("## 🤖 Agrupar (Heurístico + IA) — rápido y más barato")

    eligible = [t for t in site_tickets if eligible_for_ai(t)]
    st.caption(f"Elegibles para IA (Open + Unassigned + Requester=ServiceDesk): {len(eligible)} / {len(site_tickets)}")

    cA, cB, cC = st.columns([1, 1, 1])
    with cA:
        fuzzy_threshold = st.slider("Umbral fuzzy (heurístico)", 0.70, 0.98, 0.90, 0.01)
    with cB:
        max_candidates = st.number_input("Máx. grupos candidatos enviados a IA", min_value=5, max_value=200, value=60, step=5)
    with cC:
        show_debug = st.checkbox("Debug heurístico", value=False)

    if st.button("Agrupar (Heurístico + IA)", type="primary", disabled=len(eligible) < 2):
        try:
            # A) Heurístico
            cand = build_candidate_groups(
                eligible,
                fuzzy_threshold=float(fuzzy_threshold),
                max_candidates=int(max_candidates),
            )
            candidate_payload = []
            for g in cand:
                tickets_in_group = [compact_ticket_for_ai(t) for t in eligible if str(t.get("id")) in set(g.ids)]
                # asegurar 2+
                if len(tickets_in_group) >= 2:
                    candidate_payload.append({"reason_hint": g.reason, "tickets": tickets_in_group})

            if show_debug:
                st.write(f"Candidatos heurísticos: {len(candidate_payload)}")
                st.json(candidate_payload[:3])

            if not candidate_payload:
                st.warning("Heurístico no encontró candidatos con los filtros actuales.")
            else:
                # B) IA: solo jerarquía/padre/hijos dentro de candidatos
                ai = refine_candidate_groups_with_ai(site_name=selected_site, candidate_groups=candidate_payload)
                groups = ai.get("groups", [])

                # Guardar por sitio
                all_groups_by_site: Dict[str, List[dict]] = st.session_state.get("ai_groups_by_site", {}) or {}
                all_groups_by_site[selected_site] = groups
                st.session_state["ai_groups_by_site"] = all_groups_by_site

                st.success(f"IA devolvió {len(groups)} grupos.")

        except AIError as e:
            st.error(str(e))

    groups_by_site: Dict[str, List[dict]] = st.session_state.get("ai_groups_by_site", {}) or {}
    groups = groups_by_site.get(selected_site, []) or []

    if not groups:
        st.info("Aún no has generado grupos con IA para este sitio.")
        st.stop()

    st.markdown("### Grupos propuestos por IA (revisar antes de merge)")
    by_id = {str(t.get("id")): t for t in site_tickets}

    for idx, g in enumerate(groups, start=1):
        if not isinstance(g, dict):
            continue

        reason = str(g.get("reason", "") or "").strip()
        parent_id = str(g.get("parent_id", "") or "").strip()
        child_ids = [str(x).strip() for x in (g.get("child_ids") or []) if str(x).strip()]

        ids = []
        if parent_id:
            ids.append(parent_id)
        ids.extend([x for x in child_ids if x and x != parent_id])
        # mantener solo ids que aún existen
        ids = [x for x in ids if x in by_id]

        if len(ids) < 2:
            continue

        with st.expander(f"Grupo #{idx} ({len(ids)} tickets) — {reason}", expanded=False):
            rows = [by_id[x] for x in ids]
            st.dataframe(rows, use_container_width=True, height=220)

            default_parent = parent_id if parent_id in ids else min(ids, key=_to_int_id)

            new_parent = st.selectbox(
                "Elegir padre (ticket principal)",
                options=ids,
                index=ids.index(default_parent),
                key=f"ai_parent_sel__{selected_site}__{idx}",
            )

            selectable_children = [x for x in ids if x != new_parent]
            selected_children = st.multiselect(
                "Selecciona hijos a mergear en este padre",
                options=selectable_children,
                default=selectable_children,
                key=f"ai_children_sel__{selected_site}__{idx}",
            )

            if st.button("MERGEAR este grupo", type="primary", key=f"ai_merge_btn__{selected_site}__{idx}"):
                if not new_parent or not selected_children:
                    st.warning("Necesitas un padre y al menos 1 hijo.")
                else:
                    try:
                        _res = client.merge_requests(parent_id=new_parent, child_ids=selected_children)
                        st.success(f"Merge OK. Padre={new_parent}, hijos={selected_children}")

                        apply_merge_local_update(client, selected_site, new_parent, selected_children)
                        update_groups_after_merge(selected_site, new_parent, selected_children)

                        st.rerun()
                    except ServiceDeskApiError as e:
                        st.error(str(e))
