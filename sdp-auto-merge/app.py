from __future__ import annotations

import math
import re
import streamlit as st

from servicedesk_merge.config import load_settings, SettingsError
from servicedesk_merge.sdp_client import ServiceDeskClient, ServiceDeskApiError
from servicedesk_merge.ai_gpt import group_tickets_with_ai, AIError

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
    """
    - Quita hijos mergeados del pool general y del listado del sitio (en caliente)
    - Refresca padre desde API y lo actualiza localmente
    """
    parent_id = str(parent_id).strip()
    child_set = {str(x).strip() for x in child_ids if str(x).strip()}
    child_set.discard(parent_id)

    pool = st.session_state.get("pool_data", []) or []
    src_pool = st.session_state.get("site_source_pool", []) or pool
    site_list = st.session_state.get("site_tickets", []) or []

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
    st.session_state["site_tickets"] = site_list


def update_ai_groups_after_merge(parent_id: str, child_ids: list[str]) -> None:
    """
    Remapea grupos IA localmente sin volver a llamar GPT:
    - elimina hijos mergeados del resto de grupos
    - marca grupo como MERGED_DONE si aplica
    """
    groups = st.session_state.get("ai_groups")
    if not isinstance(groups, list) or not groups:
        return

    parent_id = str(parent_id).strip()
    merged_children = {str(x).strip() for x in child_ids if str(x).strip()}
    merged_children.discard(parent_id)

    new_groups = []
    for g in groups:
        if not isinstance(g, dict):
            continue

        reason = str(g.get("reason", "") or "")
        g_parent = str(g.get("parent_id", "") or "").strip()
        g_children = [str(x).strip() for x in (g.get("child_ids") or []) if str(x).strip()]

        # no tocamos el “compilado” aquí; lo usamos como UI manual
        if reason == "ALL_TICKETS_NO_MERGE":
            new_groups.append(g)
            continue

        g_children_filtered = [cid for cid in g_children if cid not in merged_children]

        if g_parent == parent_id:
            g2 = dict(g)
            g2["child_ids"] = g_children_filtered
            g2["merged_done"] = True
            if "MERGED_DONE" not in g2.get("reason", ""):
                g2["reason"] = f"{reason} | MERGED_DONE"
            new_groups.append(g2)
            continue

        if g_parent in merged_children:
            g2 = dict(g)
            g2["child_ids"] = g_children_filtered
            g2["merged_done"] = True
            if "PARENT_WAS_MERGED_REVIEW" not in g2.get("reason", ""):
                g2["reason"] = f"{reason} | PARENT_WAS_MERGED_REVIEW"
            new_groups.append(g2)
            continue

        g2 = dict(g)
        g2["child_ids"] = g_children_filtered
        new_groups.append(g2)

    st.session_state["ai_groups"] = new_groups


def _to_int_id(x: str) -> int:
    try:
        return int(str(x).strip())
    except Exception:
        return 10**18


def _extract_last_octets(subject: str) -> list[int]:
    ips = IP_RE.findall(subject or "")
    out = []
    for ip in ips:
        parts = ip.split(".")
        if len(parts) == 4:
            try:
                out.append(int(parts[3]))
            except Exception:
                pass
    return out


def _device_rank(ticket: dict) -> int:
    """
    Menor rank = más arriba en jerarquía (mejor candidato a padre)
    Router (0) -> Switch100/Core (1) -> Switch secundario (2) -> Radio (3) -> Switch terciario (4) -> Endpoint (5) -> Unknown (6)
    """
    subj = (ticket.get("subject") or "").lower()
    octets = _extract_last_octets(subj)

    if "router" in subj or "gateway" in subj or "gw" in subj:
        return 0

    # .10x => 100-109 (Switch 100/Core)
    if any(100 <= o <= 109 for o in octets) or "switch 100" in subj or "sw100" in subj or "core switch" in subj:
        return 1

    # Switch secundario (SM)
    if "switch" in subj or " sm" in subj or subj.startswith("sm ") or "sm down" in subj or "sm ca" in subj:
        return 2

    if "radio" in subj:
        return 3

    if "tertiary" in subj or "tercer" in subj or "access switch" in subj or "switch 3" in subj or "sw3" in subj:
        return 4

    if "camera" in subj or " cam" in subj or "ap " in subj or " ap" in subj or "nvr" in subj:
        return 5

    return 6


def normalize_groups_choose_parent(groups: list[dict], tickets: list[dict]) -> list[dict]:
    """
    Reescribe parent_id usando:
    1) Jerarquía (device_rank más bajo)
    2) Desempate: ID numéricamente menor (más viejo)
    """
    by_id = {str(t.get("id")): t for t in tickets}

    new_groups: list[dict] = []
    for g in groups or []:
        if not isinstance(g, dict):
            continue

        reason = str(g.get("reason", "") or "")
        if reason == "ALL_TICKETS_NO_MERGE":
            new_groups.append(g)
            continue

        parent_id = str(g.get("parent_id", "") or "").strip()
        child_ids = [str(x).strip() for x in (g.get("child_ids") or []) if str(x).strip()]

        ids = []
        if parent_id:
            ids.append(parent_id)
        ids.extend(child_ids)

        uniq = []
        for x in ids:
            if x and x not in uniq and x in by_id:
                uniq.append(x)

        if len(uniq) < 2:
            new_groups.append(g)
            continue

        best = None
        best_key = None
        for tid in uniq:
            t = by_id.get(tid, {})
            key = (_device_rank(t), _to_int_id(tid))
            if best is None or key < best_key:
                best = tid
                best_key = key

        new_parent = best
        new_children = [x for x in uniq if x != new_parent]

        g2 = dict(g)
        g2["parent_id"] = new_parent
        g2["child_ids"] = new_children
        new_groups.append(g2)

    return new_groups


def ensure_compiled_group(groups: list[dict], tickets: list[dict]) -> list[dict]:
    """
    Garantiza que exista un grupo especial (para UI manual) al final:
      reason = ALL_TICKETS_NO_MERGE
      parent_id = ID numéricamente menor (solo informativo)
      child_ids = []
    """
    cleaned = []
    for g in (groups or []):
        if isinstance(g, dict) and str(g.get("reason", "")).strip() != "ALL_TICKETS_NO_MERGE":
            cleaned.append(g)

    numeric_ids = []
    for t in tickets or []:
        tid = str(t.get("id", "")).strip()
        try:
            numeric_ids.append(int(tid))
        except Exception:
            pass
    parent_id = str(min(numeric_ids)) if numeric_ids else ""

    cleaned.append({"reason": "ALL_TICKETS_NO_MERGE", "parent_id": parent_id, "child_ids": []})
    return cleaned


def build_default_best_parent(all_ids: list[str], by_id: dict[str, dict]) -> str | None:
    """
    Sugiere un padre por jerarquía (y desempate por ID menor) para el tablero manual.
    """
    best = None
    best_key = None
    for tid in all_ids:
        t = by_id.get(tid, {})
        key = (_device_rank(t), _to_int_id(tid))
        if best is None or key < best_key:
            best = tid
            best_key = key
    return best


# ----------------------------- UI -----------------------------
st.set_page_config(page_title="SDP Loader + IA", layout="wide")
st.title("SDP Loader + IA")

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

tab_pool, tab_site = st.tabs(["📥 Pool general", "🏢 Por sitio + IA + merge"])

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
            st.session_state["site_tickets"] = []
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
    st.subheader("Por sitio (sin /sites) + IA + merge")
    st.info("El dropdown de sitios se deriva del pool cargado (sin llamar /sites).")

    src_pool = st.session_state.get("site_source_pool", [])
    if not src_pool:
        st.warning("Primero carga un pool en el tab 'Pool general'.")
        st.stop()

    sites = derive_sites_from_tickets(src_pool)
    if not sites:
        st.warning("No pude derivar sitios desde el pool (campo 'site' vacío).")
        st.stop()

    selected_site = st.selectbox("Selecciona un sitio", options=sites)

    # Reset plan si cambia el sitio
    if st.session_state.get("manual_plan_site") != selected_site:
        st.session_state["manual_plan_site"] = selected_site
        st.session_state["manual_plan"] = {}  # parent_id -> [child_ids]

    cA, cB = st.columns([1, 1])
    with cA:
        max_to_show = st.number_input("Máximo a mostrar en tabla", min_value=50, max_value=10000, value=2000, step=50)
    with cB:
        refresh_after_merge = st.checkbox("Refrescar padre desde API al merge", value=True)

    if st.button("Cargar tickets de este sitio", type="primary"):
        site_all = [t for t in src_pool if t.get("site") == selected_site]
        st.session_state["site_tickets"] = site_all[: int(max_to_show)]
        st.success(f"Tickets en '{selected_site}': {len(site_all)} (mostrando {len(st.session_state['site_tickets'])})")

    site_tickets = st.session_state.get("site_tickets", [])
    if not site_tickets:
        st.stop()

    st.markdown("### Tickets del sitio")
    st.dataframe(site_tickets, use_container_width=True, height=420)

    st.divider()
    st.markdown("## IA: agrupar tickets similares (posibles duplicados)")

    if st.button("Agrupar con IA"):
        try:
            ai = group_tickets_with_ai(site_tickets)
            groups = ai.get("groups", [])
            groups = normalize_groups_choose_parent(groups, site_tickets)
            groups = ensure_compiled_group(groups, site_tickets)  # para el tablero manual
            st.session_state["ai_groups"] = groups
        except AIError as e:
            st.error(str(e))

    groups = st.session_state.get("ai_groups")
    if not groups:
        st.stop()

    st.success(f"Grupos IA actuales: {len(groups)}")

    by_id = {str(t.get("id")): t for t in site_tickets}

    # ---------------- Render de grupos ----------------
    for i, g in enumerate(groups, start=1):
        if not isinstance(g, dict):
            continue

        reason = str(g.get("reason", "") or "").strip()

        # ✅ REEMPLAZO TOTAL del ALL_TICKETS_NO_MERGE: ahora es tablero manual multi-merge
        if reason == "ALL_TICKETS_NO_MERGE":
            with st.expander("📌 Compilado (todos los tickets) + Multi-merge manual", expanded=True):
                st.caption(
                    "Aquí puedes hacer merges rápidos SIN IA: elige varios padres y asigna hijos a cada uno. "
                    "Luego ejecuta todo en una sola corrida."
                )

                # Tabla completa (compilado)
                compiled = sorted(site_tickets, key=lambda t: _to_int_id(t.get("id", "")))
                st.dataframe(compiled, use_container_width=True, height=320)

                all_ids = [str(t.get("id")) for t in compiled if str(t.get("id", "")).strip()]

                if len(all_ids) < 2:
                    st.info("No hay suficientes tickets para hacer merges manuales.")
                    continue

                suggested_parent = build_default_best_parent(all_ids, by_id)

                # Selección de padres
                default_parents = [suggested_parent] if suggested_parent in all_ids else []
                parents = st.multiselect(
                    "Selecciona uno o varios PADRES",
                    options=all_ids,
                    default=default_parents,
                    key="manual_parents",
                )

                # Mantener plan en session_state
                plan: dict[str, list[str]] = st.session_state.get("manual_plan", {}) or {}

                # Limpieza: si un padre deja de estar seleccionado, lo removemos del plan
                plan = {p: kids for p, kids in plan.items() if p in parents}

                # Hijos no se pueden repetir entre padres (para evitar merges ambiguos)
                used_children = set()
                for p, kids in plan.items():
                    for k in kids:
                        used_children.add(k)

                st.markdown("### Asignación de hijos por cada padre")

                for p in parents:
                    # hijos disponibles = todos menos padres, menos el mismo padre, menos ya usados por otros padres
                    other_parents = set(parents) - {p}

                    available = [
                        x for x in all_ids
                        if x != p and x not in other_parents and x not in (used_children - set(plan.get(p, [])))
                    ]

                    current = plan.get(p, [])
                    # si current tiene algo que ya no está disponible, lo removemos
                    current = [x for x in current if x in available]

                    kids = st.multiselect(
                        f"Hijos para el padre {p}",
                        options=available,
                        default=current,
                        key=f"manual_children_{p}",
                    )

                    # actualizar used_children para el siguiente padre
                    # (primero quitamos lo anterior y añadimos lo nuevo)
                    for x in plan.get(p, []):
                        used_children.discard(x)
                    for x in kids:
                        used_children.add(x)

                    plan[p] = kids

                # Guardar plan
                st.session_state["manual_plan"] = plan

                st.markdown("### Resumen del plan de merges")
                summary_rows = []
                for p, kids in plan.items():
                    if kids:
                        summary_rows.append({"parent": p, "children_count": len(kids), "children": ", ".join(kids)})

                if summary_rows:
                    st.dataframe(summary_rows, use_container_width=True, height=180)
                else:
                    st.info("Aún no has asignado hijos a ningún padre.")

                c1, c2 = st.columns([1, 1])
                with c1:
                    if st.button("🧹 Limpiar plan", type="secondary"):
                        st.session_state["manual_plan"] = {}
                        st.session_state["manual_parents"] = []
                        st.rerun()

                with c2:
                    run_all = st.button("🚀 Ejecutar merges del plan", type="primary", disabled=not bool(summary_rows))

                if run_all:
                    # Ejecutar secuencialmente
                    # Orden recomendado: padres “más arriba” primero por jerarquía y luego por ID
                    def parent_sort_key(pid: str):
                        t = by_id.get(pid, {})
                        return (_device_rank(t), _to_int_id(pid))

                    operations = [(p, plan[p]) for p in plan.keys() if plan.get(p)]
                    operations.sort(key=lambda x: parent_sort_key(x[0]))

                    ok_count = 0
                    fail_count = 0

                    for parent_id, child_ids in operations:
                        # Si los hijos ya no existen (quizá por merges previos en esta corrida), filtramos
                        current_ids = {str(t.get("id")) for t in st.session_state.get("site_tickets", [])}
                        child_ids = [c for c in child_ids if c in current_ids and c != parent_id]

                        if not child_ids:
                            continue

                        try:
                            _res = client.merge_requests(parent_id=parent_id, child_ids=child_ids)
                            ok_count += 1

                            apply_merge_local_update(client, selected_site, parent_id, child_ids)
                            update_ai_groups_after_merge(parent_id, child_ids)

                        except ServiceDeskApiError as e:
                            fail_count += 1
                            st.error(f"Fallo merge padre={parent_id} hijos={child_ids}\n{e}")

                    st.success(f"Listo. Merges OK: {ok_count}, fallidos: {fail_count}")

                    # Limpiar plan porque ya cambió el universo de tickets
                    st.session_state["manual_plan"] = {}
                    st.session_state["manual_parents"] = []

                    # Re-render en caliente
                    st.rerun()

            continue  # no renderiza nada más para este grupo especial

        # --------- Grupos “normales” de IA ---------
        parent_id = str(g.get("parent_id", "") or "").strip()
        child_ids = [str(x).strip() for x in (g.get("child_ids") or []) if str(x).strip()]

        ids = []
        if parent_id:
            ids.append(parent_id)
        ids.extend([x for x in child_ids if x and x != parent_id])
        ids = [x for x in ids if x in by_id]

        with st.expander(f"Grupo #{i} ({len(ids)} tickets) — {reason}", expanded=False):
            rows = [by_id[x] for x in ids]
            st.dataframe(rows, use_container_width=True, height=220)

            if not ids:
                st.warning("Grupo vacío: IDs no están en la tabla actual (quizá ya fueron mergeados).")
                continue

            is_done = bool(g.get("merged_done")) or ("MERGED_DONE" in reason)

            default_parent = parent_id if parent_id in ids else ids[0]
            new_parent = st.selectbox(
                "Elegir padre (ticket principal)",
                options=ids,
                index=ids.index(default_parent),
                key=f"parent_sel_{i}",
                disabled=is_done,
            )

            selectable_children = [x for x in ids if x != new_parent]
            selected_children = st.multiselect(
                "Selecciona hijos a mergear en este padre",
                options=selectable_children,
                default=selectable_children,
                key=f"children_sel_{i}",
                disabled=is_done,
            )

            do_merge = st.button(
                "MERGEAR este grupo",
                type="primary",
                key=f"merge_btn_{i}",
                disabled=is_done,
            )

            if do_merge:
                if not new_parent or not selected_children:
                    st.warning("Necesitas un padre y al menos 1 hijo.")
                else:
                    try:
                        res = client.merge_requests(parent_id=new_parent, child_ids=selected_children)
                        st.success(f"Merge OK. Padre={new_parent}, hijos={selected_children}")
                        st.json(res)

                        apply_merge_local_update(client, selected_site, new_parent, selected_children)

                        if not refresh_after_merge:
                            pass

                        update_ai_groups_after_merge(new_parent, selected_children)

                        st.rerun()

                    except ServiceDeskApiError as e:
                        st.error(str(e))
