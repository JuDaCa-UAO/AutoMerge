from __future__ import annotations

import math
import re
import streamlit as st

from servicedesk_merge.config import load_settings, SettingsError
from servicedesk_merge.sdp_client import ServiceDeskClient, ServiceDeskApiError
from servicedesk_merge.ai_gpt import group_tickets_with_ai, AIError

IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
UNASSIGNED_VALUES = {"", "unassigned", "none", "null", "-", "n/a", "na"}


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
    if "switch" in subj or " sm" in subj or subj.startswith("sm ") or "sm down" in subj or "sig-sm" in subj:
        return 2

    if "radio" in subj:
        return 3

    if "tertiary" in subj or "tercer" in subj or "access switch" in subj or "switch 3" in subj or "sw3" in subj:
        return 4

    if "camera" in subj or " cam" in subj or "ap " in subj or " ap" in subj or "nvr" in subj or "da " in subj:
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


def _norm(s: str) -> str:
    return str(s or "").strip().lower()


def is_open(ticket: dict) -> bool:
    return _norm(ticket.get("status", "")) == "open"


def is_unassigned(ticket: dict) -> bool:
    tech = _norm(ticket.get("technician", ""))
    return tech in UNASSIGNED_VALUES


def is_merge_eligible(ticket: dict) -> bool:
    return is_open(ticket) and is_unassigned(ticket)


def split_tickets_eligible(tickets: list[dict]) -> tuple[list[dict], list[dict]]:
    eligible = [t for t in tickets if is_merge_eligible(t)]
    non_eligible = [t for t in tickets if not is_merge_eligible(t)]
    return eligible, non_eligible


def render_manual_multimerge_board(*, client: ServiceDeskClient, selected_site: str, site_tickets: list[dict]) -> None:
    """
    Tablero manual SIEMPRE visible:
    - Por defecto: Open + Unassigned (lo que más te interesa)
    - Opcional: incluir technician asignado (override)
    - Selección rápida por tabla: seleccionar filas y usar botones para setear padre / añadir hijos
    - Multi-merge: varios padres y asignar hijos por padre
    """

    st.markdown("## 📌 Compilado + Multi-merge manual (sin IA)")

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        open_only = st.checkbox("Solo status Open", value=True, key=f"mm_open_only__{selected_site}")
    with c2:
        include_assigned = st.checkbox(
            "Incluir tickets con technician asignado (override)",
            value=False,
            key=f"mm_include_assigned__{selected_site}",
        )
    with c3:
        st.caption("Tip: selecciona filas en la tabla y usa los botones para PADRE/HIJOS.")

    if include_assigned:
        st.warning(
            "Override activo: estás incluyendo tickets con technician asignado. "
            "Úsalo solo cuando estés seguro."
        )
        confirm_override = st.checkbox(
            "Confirmo que quiero permitir merges incluyendo tickets asignados",
            value=False,
            key=f"mm_confirm_override__{selected_site}",
        )
    else:
        confirm_override = True

    # ---- construir universo para merge manual ----
    def _ok_open(t: dict) -> bool:
        return (not open_only) or (_norm(t.get("status", "")) == "open")

    def _ok_assigned(t: dict) -> bool:
        if include_assigned:
            return True
        return is_unassigned(t)

    merge_universe = [t for t in site_tickets if _ok_open(t) and _ok_assigned(t)]
    info_nonmerge = [t for t in site_tickets if t not in merge_universe]

    # ---- tablas ----
    if not merge_universe:
        st.warning("No hay tickets que cumplan el filtro actual para merge manual.")
    else:
        compiled = sorted(merge_universe, key=lambda t: _to_int_id(t.get("id", "")))
        st.dataframe(
            compiled,
            use_container_width=True,
            height=340,
            selection_mode="multi-row",
            on_select="rerun",
            key=f"mm_table__{selected_site}",
        )

    with st.expander("ℹ️ Informativo (fuera del filtro actual)", expanded=False):
        st.caption(
            "Estos tickets NO están en el universo actual de merge (por status o technician). "
            "Útiles para ver si un incidente automático ya lo está manejando alguien."
        )
        if info_nonmerge:
            st.dataframe(
                sorted(info_nonmerge, key=lambda t: _to_int_id(t.get("id", ""))),
                use_container_width=True,
                height=240,
            )
        else:
            st.info("Nada fuera del filtro actual.")

    if len(merge_universe) < 2:
        return

    all_ids = [str(t.get("id")) for t in merge_universe if str(t.get("id", "")).strip()]
    by_id = {str(t.get("id")): t for t in merge_universe}

    # ---- mantener plan ----
    plan_key = f"manual_plan__{selected_site}"
    parents_key = f"manual_parents__{selected_site}"
    active_parent_key = f"manual_active_parent__{selected_site}"

    plan: dict[str, list[str]] = st.session_state.get(plan_key, {}) or {}

    # sugerencia de padre inicial
    suggested_parent = build_default_best_parent(all_ids, by_id)
    default_parents = [suggested_parent] if suggested_parent in all_ids else []

    parents = st.multiselect(
        "Selecciona uno o varios PADRES",
        options=sorted(all_ids, key=_to_int_id),
        default=default_parents,
        key=parents_key,
    )

    # limpiar plan de padres no presentes
    plan = {p: kids for p, kids in plan.items() if p in parents}

    if parents:
        # padre activo (sobre el que “trabajas” al añadir hijos)
        if st.session_state.get(active_parent_key) not in parents:
            st.session_state[active_parent_key] = parents[0]

        active_parent = st.selectbox(
            "Padre activo (los seleccionados se agregan como hijos aquí)",
            options=parents,
            index=parents.index(st.session_state.get(active_parent_key, parents[0])),
            key=active_parent_key,
        )
    else:
        active_parent = None

    # ---- leer selección de filas de la tabla ----
    selected_ids: list[str] = []
    sel_state = st.session_state.get(f"mm_table__{selected_site}")
    # En Streamlit, la selección queda en sel_state["selection"]["rows"] o sel_state.selection.rows
    try:
        rows_idx = sel_state.selection.rows  # type: ignore[attr-defined]
    except Exception:
        rows_idx = None

    if rows_idx is None:
        # fallback (según versión)
        try:
            rows_idx = (sel_state.get("selection") or {}).get("rows")
        except Exception:
            rows_idx = []

    if rows_idx and merge_universe:
        # la tabla muestra `compiled` (ordenado). reconstruimos mismo orden:
        compiled = sorted(merge_universe, key=lambda t: _to_int_id(t.get("id", "")))
        for idx in rows_idx:
            if 0 <= int(idx) < len(compiled):
                selected_ids.append(str(compiled[int(idx)].get("id")))

    # ---- Acciones rápidas: padre / hijos usando selección ----
    st.markdown("### Acciones rápidas (usando selección de la tabla)")

    a1, a2, a3, a4 = st.columns([1, 1, 1, 2])
    with a1:
        btn_set_parent = st.button("⭐ Usar seleccionado como PADRE", disabled=(len(selected_ids) != 1))
    with a2:
        btn_add_children = st.button("➕ Agregar seleccionados como HIJOS", disabled=(not active_parent or not selected_ids))
    with a3:
        btn_remove_children = st.button("➖ Quitar seleccionados de HIJOS", disabled=(not active_parent or not selected_ids))
    with a4:
        st.caption(f"Seleccionados: {', '.join(selected_ids) if selected_ids else '(nada)'}")

    if btn_set_parent:
        # cambia el padre activo y lo añade a lista de padres si no estaba
        pid = selected_ids[0]
        if pid not in parents:
            parents = [pid] + parents
            st.session_state[parents_key] = parents
        st.session_state[active_parent_key] = pid
        st.rerun()

    if btn_add_children and active_parent:
        kids = set(plan.get(active_parent, []))
        for sid in selected_ids:
            if sid == active_parent:
                continue
            # no permitir que un hijo sea padre de otro a la vez
            if sid in parents:
                continue
            kids.add(sid)
        plan[active_parent] = sorted(kids, key=_to_int_id)
        st.session_state[plan_key] = plan
        st.rerun()

    if btn_remove_children and active_parent:
        kids = set(plan.get(active_parent, []))
        for sid in selected_ids:
            kids.discard(sid)
        plan[active_parent] = sorted(kids, key=_to_int_id)
        st.session_state[plan_key] = plan
        st.rerun()

    # ---- UI de asignación tradicional por padre (opcional, para ver/editar rápido) ----
    with st.expander("Editar plan manual (vista detallada)", expanded=False):
        used_children = set()
        for p, kids in plan.items():
            for k in kids:
                used_children.add(k)

        for p in parents:
            other_parents = set(parents) - {p}
            available = [
                x
                for x in all_ids
                if x != p and x not in other_parents and x not in (used_children - set(plan.get(p, [])))
            ]
            current = [x for x in plan.get(p, []) if x in available]

            kids = st.multiselect(
                f"Hijos para el padre {p}",
                options=sorted(available, key=_to_int_id),
                default=sorted(current, key=_to_int_id),
                key=f"manual_children__{selected_site}__{p}",
            )
            plan[p] = kids

        st.session_state[plan_key] = plan

    # ---- Resumen ----
    st.markdown("### Resumen del plan de merges")
    summary_rows = []
    for p, kids in plan.items():
        if kids:
            summary_rows.append({"parent": p, "children_count": len(kids), "children": ", ".join(sorted(kids, key=_to_int_id))})

    if summary_rows:
        st.dataframe(summary_rows, use_container_width=True, height=180)
    else:
        st.info("Aún no has asignado hijos a ningún padre.")

    cL, cR = st.columns([1, 1])
    with cL:
        if st.button("🧹 Limpiar plan", type="secondary", key=f"manual_clear__{selected_site}"):
            st.session_state[plan_key] = {}
            st.session_state[parents_key] = []
            st.session_state[active_parent_key] = None
            st.rerun()

    with cR:
        run_all = st.button(
            "🚀 Ejecutar merges del plan",
            type="primary",
            key=f"manual_run_all__{selected_site}",
            disabled=(not bool(summary_rows) or not confirm_override),
        )

    if run_all:
        def parent_sort_key(pid: str):
            t = by_id.get(pid, {})
            return (_device_rank(t), _to_int_id(pid))

        operations = [(p, plan[p]) for p in plan.keys() if plan.get(p)]
        operations.sort(key=lambda x: parent_sort_key(x[0]))

        ok_count = 0
        fail_count = 0

        # universo actual (por si cambió)
        current_ids = {str(t.get("id")) for t in merge_universe}

        for parent_id, child_ids in operations:
            # filtrar niños existentes y válidos
            child_ids = [c for c in child_ids if c in current_ids and c != parent_id]
            if not child_ids:
                continue

            try:
                client.merge_requests(parent_id=parent_id, child_ids=child_ids)
                ok_count += 1

                apply_merge_local_update(client, selected_site, parent_id, child_ids)
                update_ai_groups_after_merge(parent_id, child_ids)

                # actualizar universo para evitar re-merge
                for c in child_ids:
                    current_ids.discard(c)

            except ServiceDeskApiError as e:
                fail_count += 1
                st.error(f"Fallo merge padre={parent_id} hijos={child_ids}\n{e}")

        st.success(f"Listo. Merges OK: {ok_count}, fallidos: {fail_count}")

        # limpiar plan tras ejecutar
        st.session_state[plan_key] = {}
        st.session_state[parents_key] = []
        st.session_state[active_parent_key] = None
        st.rerun()



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
            st.session_state["ai_groups"] = []
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
    st.subheader("Por sitio + merge manual (siempre) + IA (opcional)")
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

    cA, cB, cC = st.columns([1, 1, 1])
    with cA:
        max_to_show = st.number_input("Máximo a mostrar en tabla", min_value=50, max_value=10000, value=2000, step=50)
    with cB:
        show_site_table = st.checkbox("Mostrar tabla completa del sitio", value=True)
    with cC:
        refresh_now = st.button("🔄 Refrescar lista del sitio")

    # Auto-carga al seleccionar sitio (y también si presionas refrescar)
    site_all = [t for t in src_pool if t.get("site") == selected_site]
    if refresh_now or st.session_state.get("site_loaded_for") != selected_site:
        st.session_state["site_loaded_for"] = selected_site
        st.session_state["site_tickets"] = site_all[: int(max_to_show)]
        st.session_state["ai_groups"] = []  # grupos IA dependen del universo actual (pero NO llamamos IA)

    site_tickets = st.session_state.get("site_tickets", [])
    if not site_tickets:
        st.warning("No hay tickets en este sitio dentro del pool actual.")
        st.stop()

    st.success(f"Tickets en '{selected_site}': {len(site_all)} (mostrando {len(site_tickets)})")

    if show_site_table:
        st.markdown("### Tickets del sitio (tal cual vienen del pool)")
        st.dataframe(site_tickets, use_container_width=True, height=420)

    # ✅ SIEMPRE mostrar primero el tablero manual (compilado) sin necesidad de IA
    st.divider()
    render_manual_multimerge_board(client=client, selected_site=selected_site, site_tickets=site_tickets)

    # IA opcional
    st.divider()
    st.markdown("## IA (opcional): agrupar tickets similares (solo Open + Unassigned)")

    eligible, _non = split_tickets_eligible(site_tickets)
    st.caption(f"Elegibles para IA: {len(eligible)} (Open + Unassigned).")

    if st.button("Agrupar con IA", type="primary"):
        if len(eligible) < 2:
            st.warning("No hay suficientes tickets elegibles (Open + Unassigned) para agrupar con IA.")
        else:
            try:
                ai = group_tickets_with_ai(eligible)
                groups = ai.get("groups", [])
                groups = normalize_groups_choose_parent(groups, eligible)
                st.session_state["ai_groups"] = groups
                st.success(f"IA generó {len(groups)} grupos (solo elegibles).")
            except AIError as e:
                st.error(str(e))

    groups = st.session_state.get("ai_groups") or []
    if not groups:
        st.stop()

    st.success(f"Grupos IA actuales: {len(groups)}")

    by_id = {str(t.get("id")): t for t in eligible}

    for i, g in enumerate(groups, start=1):
        if not isinstance(g, dict):
            continue

        reason = str(g.get("reason", "") or "").strip()
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
                st.warning("Grupo vacío: IDs no están en el universo elegible (quizá ya fueron mergeados).")
                continue

            is_done = bool(g.get("merged_done")) or ("MERGED_DONE" in reason)

            default_parent = parent_id if parent_id in ids else ids[0]
            new_parent = st.selectbox(
                "Elegir padre (ticket principal)",
                options=ids,
                index=ids.index(default_parent),
                key=f"ai_parent_sel_{i}",
                disabled=is_done,
            )

            selectable_children = [x for x in ids if x != new_parent]
            selected_children = st.multiselect(
                "Selecciona hijos a mergear en este padre",
                options=selectable_children,
                default=selectable_children,
                key=f"ai_children_sel_{i}",
                disabled=is_done,
            )

            do_merge = st.button(
                "MERGEAR este grupo",
                type="primary",
                key=f"ai_merge_btn_{i}",
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
                        update_ai_groups_after_merge(new_parent, selected_children)

                        # ✅ al hacer merge, el tablero manual (arriba) se actualiza automáticamente porque usa site_tickets en session_state
                        st.rerun()

                    except ServiceDeskApiError as e:
                        st.error(str(e))
