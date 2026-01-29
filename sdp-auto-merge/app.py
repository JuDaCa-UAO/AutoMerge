from __future__ import annotations

import math
import time
import re
from collections import defaultdict
from difflib import SequenceMatcher

import streamlit as st

from servicedesk_merge.config import load_settings, SettingsError
from servicedesk_merge.sdp_client import ServiceDeskClient, ServiceDeskApiError
from servicedesk_merge.ai_gpt import group_tickets_with_ai, AIError

IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
CAM_RANGE_RE = re.compile(r"\b(cam(?:era)?s?)\s*([0-9]{1,3})(?:\s*[-–]\s*([0-9]{1,3}))?\b", re.I)


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


def format_reason(reason: str, lang: str) -> str:
    if not reason:
        return ""
    labels_es = {
        "EXACT_SUBJECT": "Mismo asunto",
        "SAME_IP": "Misma IP",
        "SIMILAR_SUBJECT": "Asunto similar",
        "CAMERA_REF_BY_SWITCH": "Cámaras referenciadas por el switch",
        "CAMERA_REF_BY_SM": "Cámaras del mismo SM",
        "FALLBACK_EXACT_SUBJECT": "Mismo asunto (regla local)",
        "MERGED_DONE": "Merge realizado",
        "PARENT_WAS_MERGED_REVIEW": "El padre se mezcló antes",
        "POST_SPLIT_SM_CAM": "Separado por tipos de equipo",
    }
    labels_en = {
        "EXACT_SUBJECT": "Same subject",
        "SAME_IP": "Same IP",
        "SIMILAR_SUBJECT": "Similar subject",
        "CAMERA_REF_BY_SWITCH": "Cameras referenced by switch",
        "CAMERA_REF_BY_SM": "Cameras from the same SM",
        "FALLBACK_EXACT_SUBJECT": "Same subject (local rule)",
        "MERGED_DONE": "Merge completed",
        "PARENT_WAS_MERGED_REVIEW": "Parent was merged before",
        "POST_SPLIT_SM_CAM": "Split by device type",
    }
    labels = labels_es if lang == "es" else labels_en
    parts = [p.strip() for p in reason.split("|") if p.strip()]
    out = []
    for p in parts:
        if p.startswith("FALLBACK_SAME_IP:"):
            ip = p.split(":", 1)[1]
            if lang == "es":
                out.append(f"Misma IP (regla local: {ip})")
            else:
                out.append(f"Same IP (local rule: {ip})")
            continue
        out.append(labels.get(p, p))
    return " | ".join(out)


def ticket_label(by_id: dict[str, dict], tid: str, max_len: int = 80) -> str:
    t = by_id.get(tid, {})
    subject = str(t.get("subject", "")).strip()
    short = subject[:max_len] + ("..." if len(subject) > max_len else "")
    return f"{tid} - {short}" if short else tid


def start_merge_lock() -> None:
    st.session_state["merge_in_progress"] = True
    st.session_state["merge_started_at"] = time.time()


def clear_merge_lock_if_stale(max_seconds: int = 120) -> None:
    started = st.session_state.get("merge_started_at")
    if not started:
        return
    try:
        if time.time() - float(started) > max_seconds:
            st.session_state["merge_in_progress"] = False
            st.session_state["merge_started_at"] = None
    except Exception:
        st.session_state["merge_in_progress"] = False
        st.session_state["merge_started_at"] = None


def fetch_pool(client: ServiceDeskClient, pool_size: int, page_size: int, start_index: int) -> list[dict]:
    fetched = 0
    cursor = int(start_index)
    est_pages = int(math.ceil(int(pool_size) / int(page_size)))
    prog = st.progress(0, text=t("progress_start"))

    out: list[dict] = []
    for p in range(est_pages):
        batch = min(int(page_size), int(pool_size) - fetched)
        prog.progress(
            min(1.0, (p / max(est_pages, 1))),
            text=t("progress_page").format(page=p + 1, total=est_pages),
        )

        items, more, _payload = client.list_requests_page(start_index=cursor, row_count=batch)
        if not items:
            break

        out.extend([summarize_ticket(x) for x in items])
        fetched += len(items)

        if fetched >= int(pool_size) or not more:
            break
        cursor += batch

    prog.progress(1.0, text=t("progress_done").format(count=len(out)))
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


def apply_merge_local_update(
    client: ServiceDeskClient,
    site_name: str,
    parent_id: str,
    child_ids: list[str],
    *,
    refresh_parent: bool = True,
) -> None:
    parent_id = str(parent_id).strip()
    child_set = {str(x).strip() for x in child_ids if str(x).strip()}
    child_set.discard(parent_id)

    pool = st.session_state.get("pool_data", []) or []
    src_pool = st.session_state.get("site_source_pool", []) or pool
    site_list = st.session_state.get("site_tickets", []) or []

    pool = remove_ids_from_list(pool, child_set)
    src_pool = remove_ids_from_list(src_pool, child_set)
    site_list = remove_ids_from_list(site_list, child_set)

    if refresh_parent:
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


def _extract_ips(subject: str) -> set[str]:
    return set(IP_RE.findall(subject or ""))


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


def _infer_device_type(ticket: dict) -> str:
    """
    Clasificador simple por subject + IP rules.
    Tipos: router, sw_core, sw_secondary, radio, ap, sm, sw_tertiary, camera, nvr, da, unknown
    """
    subj = (ticket.get("subject") or "").lower()
    ips = _extract_ips(subj)
    octets = _extract_last_octets(subj)

    # IP-specific
    if "192.168.1.254" in ips or "router" in subj or "gateway" in subj or " gw" in subj:
        return "router"

    # NVR
    if "192.168.1.103" in ips or "nvr" in subj:
        return "nvr"

    # DA (Digital Acoustics)
    if any(ip.startswith("192.168.1.20") for ip in ips) or "digital acoustics" in subj or " da" in subj:
        return "da"

    # Switch 100/Core: .10x last octet 100-109
    if any(100 <= o <= 109 for o in octets) or "switch 100" in subj or "sw100" in subj or "core switch" in subj:
        return "sw_core"

    # Camera keywords
    if "camera" in subj or re.search(r"\bcam\b", subj):
        return "camera"

    # Radio
    if "radio" in subj:
        return "radio"

    # AP/SM IP inference on 192.168.0.*
    # AP ~ last octet ends with 0 and between 10-99 (10,20,30,...)
    if any(ip.startswith("192.168.0.") for ip in ips):
        for o in octets:
            if 10 <= o <= 99:
                if o % 10 == 0:
                    return "ap"
                if o % 10 == 1:
                    return "sm"

    # Text keywords
    if re.search(r"(^|\s)ap(\s|$)", subj):
        return "ap"
    if re.search(r"(^|\s)sm(\s|$)", subj):
        return "sm"

    # Switches (generic)
    if "switch" in subj or re.search(r"\bsw\b", subj):
        # sin más contexto lo tratamos como secundario (distribución)
        return "sw_secondary"

    # Tertiary hints
    if "tertiary" in subj or "tercer" in subj or "access switch" in subj or "switch 3" in subj or "sw3" in subj:
        return "sw_tertiary"

    return "unknown"


def _device_rank(ticket: dict) -> int:
    """
    Menor rank = más arriba (mejor candidato a padre)

    Jerarquía ajustada (como pediste):
    Router -> Switch100/Core -> Switch secundario -> Radio -> AP -> SM -> Switch terciario -> Endpoints (Cam/NVR/DA) -> Unknown
    """
    t = _infer_device_type(ticket)
    order = {
        "router": 0,
        "sw_core": 1,
        "sw_secondary": 2,
        "radio": 3,
        "ap": 4,
        "sm": 5,
        "sw_tertiary": 6,
        "camera": 7,
        "nvr": 7,
        "da": 7,
        "unknown": 8,
    }
    return order.get(t, 8)


def normalize_groups_choose_parent(groups: list[dict], tickets: list[dict]) -> list[dict]:
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

    cleaned.append(
        {
            "reason": "ALL_TICKETS_NO_MERGE",
            "parent_id": parent_id,
            "child_ids": [],
            "confidence": 1.0,
            "needs_review": False,
            "evidence": ["all_tickets_board"],
        }
    )
    return cleaned


def build_default_best_parent(all_ids: list[str], by_id: dict[str, dict]) -> str | None:
    best = None
    best_key = None
    for tid in all_ids:
        t = by_id.get(tid, {})
        key = (_device_rank(t), _to_int_id(tid))
        if best is None or key < best_key:
            best = tid
            best_key = key
    return best


def _eligible_for_ai_merge(t: dict) -> bool:
    status = (t.get("status") or "").strip().lower()
    technician = (t.get("technician") or "").strip().lower()
    requester = (t.get("requester") or "").strip()

    is_open = status == "open"
    tech_unassigned = technician in ("", "unassigned", "none", "-", "null")
    req_ok = requester == "ServiceDesk"
    return is_open and tech_unassigned and req_ok


def _has_explicit_cam_reference(sm_or_ap_subject: str) -> bool:
    # basta con que el subject mencione cam/cameras con números/rangos
    return bool(CAM_RANGE_RE.search(sm_or_ap_subject or ""))


def _extract_cam_numbers(subject: str) -> set[int]:
    """
    Extrae numeros de camara desde patrones como:
    "Cam 12", "Camera 10-12", "Cams 3-5"
    """
    out: set[int] = set()
    for _m in CAM_RANGE_RE.finditer(subject or ""):
        try:
            start = int(_m.group(2))
        except Exception:
            continue
        end_raw = _m.group(3)
        if end_raw:
            try:
                end = int(end_raw)
            except Exception:
                end = start
            if end < start:
                start, end = end, start
            for n in range(start, end + 1):
                out.add(n)
        else:
            out.add(start)
    return out


def _camera_numbers_from_ticket(ticket: dict) -> set[int]:
    nums = set(_extract_cam_numbers(ticket.get("subject", "") or ""))
    for ip in _camera_ips_from_subject(ticket.get("subject", "") or ""):
        parts = ip.split(".")
        if len(parts) == 4:
            try:
                nums.add(int(parts[3]))
            except Exception:
                pass
    return nums


def _camera_ips_from_subject(subject: str) -> set[str]:
    ips = _extract_ips(subject)
    return {ip for ip in ips if ip.startswith("192.168.1.") or ip.startswith("192.168.2.")}


def post_validate_ai_groups(groups: list[dict], tickets: list[dict]) -> list[dict]:
    """
    Barandas duras:
    - evita SM/AP con Cam/NVR/DA por coincidencia de número
    - solo permite cruzar si comparten IP completa o SM/AP menciona explícitamente cams en su subject
    - evita que un ticket aparezca en múltiples grupos
    - elimina grupos <2
    """
    by_id = {str(t.get("id")): t for t in tickets}
    used: set[str] = set()
    out: list[dict] = []

    for g in groups or []:
        if not isinstance(g, dict):
            continue

        reason = str(g.get("reason", "") or "").strip()
        if reason == "ALL_TICKETS_NO_MERGE":
            out.append(g)
            continue

        parent_id = str(g.get("parent_id", "") or "").strip()
        child_ids = [str(x).strip() for x in (g.get("child_ids") or []) if str(x).strip()]

        ids = []
        if parent_id:
            ids.append(parent_id)
        ids.extend([x for x in child_ids if x and x != parent_id])

        # limpiar inexistentes
        ids = [x for x in ids if x in by_id]

        # evitar que un ticket esté en más de un grupo
        ids = [x for x in ids if x not in used]

        if len(ids) < 2:
            continue

        # detectar tipos
        types = {tid: _infer_device_type(by_id[tid]) for tid in ids}
        has_sm_ap = any(types[tid] in ("sm", "ap") for tid in ids)
        has_cam_family = any(types[tid] in ("camera", "nvr", "da") for tid in ids)

        if has_sm_ap and has_cam_family:
            # permitir cruce solo con evidencia fuerte:
            # A) IP completa compartida con alguien en el grupo
            # B) subject del SM/AP menciona explícitamente cams
            ips_map = {tid: _extract_ips(by_id[tid].get("subject", "")) for tid in ids}

            # IP compartida (al menos 1 ip en común entre tickets)
            shared_ip_tickets = set()
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    a, b = ids[i], ids[j]
                    if ips_map[a] and ips_map[b] and (ips_map[a] & ips_map[b]):
                        shared_ip_tickets.add(a)
                        shared_ip_tickets.add(b)

            explicit_cam_ok = set()
            for tid in ids:
                if types[tid] in ("sm", "ap"):
                    if _has_explicit_cam_reference(by_id[tid].get("subject", "")):
                        explicit_cam_ok.add(tid)

            # tickets permitidos:
            # - si es sm/ap, solo puede quedarse si tiene shared ip o explicit cam reference
            # - si es cam/nvr/da, solo puede quedarse si comparte ip con alguien (no por número)
            filtered = []
            for tid in ids:
                ttype = types[tid]
                if ttype in ("sm", "ap"):
                    if tid in shared_ip_tickets or tid in explicit_cam_ok:
                        filtered.append(tid)
                elif ttype in ("camera", "nvr", "da"):
                    if tid in shared_ip_tickets:
                        filtered.append(tid)
                else:
                    # switches/radio: se mantienen, pero no “conectan” por número
                    filtered.append(tid)

            # Si al filtrar se rompe el grupo, lo partimos por familias (sm/ap vs cam family) sin cruzar
            if len(filtered) < 2:
                # split por tipo
                smap = [tid for tid in ids if types[tid] in ("sm", "ap", "sw_secondary", "sw_core", "sw_tertiary", "radio", "router")]
                camfam = [tid for tid in ids if types[tid] in ("camera", "nvr", "da")]

                # mantenemos cada subgrupo si tiene >=2
                for sub in (smap, camfam):
                    sub = [x for x in sub if x not in used]
                    if len(sub) >= 2:
                        # padre por ranking + id menor
                        best = min(sub, key=lambda x: (_device_rank(by_id[x]), _to_int_id(x)))
                        children = [x for x in sub if x != best]
                        g2 = dict(g)
                        g2["parent_id"] = best
                        g2["child_ids"] = children
                        g2["reason"] = f"{reason} | POST_SPLIT_SM_CAM"
                        out.append(g2)
                        used.update(sub)
                continue

            ids = filtered

        if len(ids) < 2:
            continue

        # recomputar padre con jerarquía local (si IA se equivocó)
        best = min(ids, key=lambda x: (_device_rank(by_id[x]), _to_int_id(x)))
        children = [x for x in ids if x != best]

        g2 = dict(g)
        g2["parent_id"] = best
        g2["child_ids"] = children
        out.append(g2)
        used.update(ids)

    return out


def fallback_groups_no_ai(site_tickets: list[dict]) -> list[dict]:
    """
    Si IA devuelve vacío: generamos grupos básicos:
    - exact subject normalizado
    - misma IP completa
    """
    by_id = {str(t.get("id")): t for t in site_tickets}
    norm = lambda s: " ".join((s or "").strip().lower().split())

    buckets_exact = defaultdict(list)
    buckets_ip = defaultdict(list)

    for t in site_tickets:
        tid = str(t.get("id", "")).strip()
        subj = str(t.get("subject", "")).strip()
        if not tid or not subj:
            continue
        buckets_exact[norm(subj)].append(tid)
        for ip in _extract_ips(subj):
            buckets_ip[ip].append(tid)

    groups = []

    # exact subject
    for k, ids in buckets_exact.items():
        if len(ids) >= 2:
            best = min(ids, key=lambda x: (_device_rank(by_id[x]), _to_int_id(x)))
            children = [x for x in ids if x != best]
            groups.append(
                {
                    "reason": "FALLBACK_EXACT_SUBJECT",
                    "parent_id": best,
                    "child_ids": children,
                    "confidence": 0.75,
                    "needs_review": True,
                    "evidence": ["exact_subject_fallback"],
                }
            )

    # same IP (solo si no está ya incluido por exact)
    used_ids = set()
    for g in groups:
        used_ids.add(g["parent_id"])
        used_ids.update(g["child_ids"])

    for ip, ids in buckets_ip.items():
        ids = [x for x in ids if x not in used_ids]
        if len(ids) >= 2:
            best = min(ids, key=lambda x: (_device_rank(by_id[x]), _to_int_id(x)))
            children = [x for x in ids if x != best]
            groups.append(
                {
                    "reason": f"FALLBACK_SAME_IP:{ip}",
                    "parent_id": best,
                    "child_ids": children,
                    "confidence": 0.8,
                    "needs_review": True,
                    "evidence": [f"same_ip_fallback:{ip}"],
                }
            )

    return groups


def groups_from_switch_camera_reference(
    site_tickets: list[dict],
    existing_groups: list[dict],
) -> list[dict]:
    """
    Si un switch menciona camaras en su subject, agrupa esas camaras con el switch.
    Solo agrega grupos nuevos sin reutilizar tickets ya usados.
    """
    by_id = {str(t.get("id")): t for t in site_tickets}
    used: set[str] = set()

    camera_by_num: dict[int, list[str]] = defaultdict(list)
    camera_by_ip: dict[str, list[str]] = defaultdict(list)
    for t in site_tickets:
        tid = str(t.get("id", "")).strip()
        if not tid:
            continue
        if _infer_device_type(t) != "camera":
            continue
        nums = _camera_numbers_from_ticket(t)
        for n in nums:
            camera_by_num[n].append(tid)
        for ip in _camera_ips_from_subject(t.get("subject", "") or ""):
            camera_by_ip[ip].append(tid)

    groups: list[dict] = []
    for t in site_tickets:
        tid = str(t.get("id", "")).strip()
        if not tid or tid in used:
            continue
        ttype = _infer_device_type(t)
        if ttype not in ("sw_core", "sw_secondary", "sw_tertiary"):
            continue
        cam_nums = _extract_cam_numbers(t.get("subject", "") or "")
        cam_ips = _camera_ips_from_subject(t.get("subject", "") or "")
        if not cam_nums and not cam_ips:
            continue

        related: list[str] = []
        for n in cam_nums:
            related.extend(camera_by_num.get(n, []))
        for ip in cam_ips:
            related.extend(camera_by_ip.get(ip, []))
        related = [x for x in dict.fromkeys(related) if x not in used and x != tid]

        ids = [tid] + related
        if len(ids) < 2:
            continue

        parent = min(ids, key=lambda x: (_device_rank(by_id.get(x, {})), _to_int_id(x)))
        children = [x for x in ids if x != parent]
        groups.append(
            {
                "reason": "CAMERA_REF_BY_SWITCH",
                "parent_id": parent,
                "child_ids": children,
                "confidence": 0.7,
                "needs_review": True,
                "evidence": ["switch_camera_reference"],
            }
        )
        used.update(ids)

    return groups


def _filter_groups_without_overlap(groups: list[dict], used: set[str]) -> list[dict]:
    out: list[dict] = []
    for g in groups or []:
        if not isinstance(g, dict):
            continue
        reason = str(g.get("reason", "") or "").strip()
        if reason == "ALL_TICKETS_NO_MERGE":
            out.append(g)
            continue
        parent_id = str(g.get("parent_id", "") or "").strip()
        child_ids = [str(x).strip() for x in (g.get("child_ids") or []) if str(x).strip()]

        ids = []
        if parent_id:
            ids.append(parent_id)
        ids.extend([x for x in child_ids if x and x != parent_id])
        ids = [x for x in ids if x and x not in used]

        if len(ids) < 2:
            continue

        new_parent = ids[0] if parent_id not in ids else parent_id
        new_children = [x for x in ids if x != new_parent]
        g2 = dict(g)
        g2["parent_id"] = new_parent
        g2["child_ids"] = new_children
        out.append(g2)
        used.update(ids)
    return out


def groups_from_sm_camera_reference(
    site_tickets: list[dict],
    existing_groups: list[dict],
) -> list[dict]:
    """
    Agrupa camaras pertenecientes a un mismo SM cuando el subject del SM las menciona.
    """
    by_id = {str(t.get("id")): t for t in site_tickets}
    used: set[str] = set()

    camera_by_num: dict[int, list[str]] = defaultdict(list)
    camera_by_ip: dict[str, list[str]] = defaultdict(list)
    for t in site_tickets:
        tid = str(t.get("id", "")).strip()
        if not tid:
            continue
        if _infer_device_type(t) != "camera":
            continue
        nums = _camera_numbers_from_ticket(t)
        for n in nums:
            camera_by_num[n].append(tid)
        for ip in _camera_ips_from_subject(t.get("subject", "") or ""):
            camera_by_ip[ip].append(tid)

    groups: list[dict] = []
    for t in site_tickets:
        tid = str(t.get("id", "")).strip()
        if not tid or tid in used:
            continue
        if _infer_device_type(t) != "sm":
            continue

        cam_nums = _extract_cam_numbers(t.get("subject", "") or "")
        cam_ips = _camera_ips_from_subject(t.get("subject", "") or "")
        if not cam_nums and not cam_ips:
            continue

        related: list[str] = []
        for n in cam_nums:
            related.extend(camera_by_num.get(n, []))
        for ip in cam_ips:
            related.extend(camera_by_ip.get(ip, []))
        related = [x for x in dict.fromkeys(related) if x not in used and x != tid]

        ids = [tid] + related
        if len(ids) < 2:
            continue

        parent = min(ids, key=lambda x: (_device_rank(by_id.get(x, {})), _to_int_id(x)))
        children = [x for x in ids if x != parent]
        groups.append(
            {
                "reason": "CAMERA_REF_BY_SM",
                "parent_id": parent,
                "child_ids": children,
                "confidence": 0.75,
                "needs_review": True,
                "evidence": ["sm_camera_reference"],
            }
        )
        used.update(ids)

    return groups


# ----------------------------- UI -----------------------------
LANG = {
    "es": {
        "app_title": "SDP Auto Merge",
        "tab_pool": "Pool general",
        "tab_site": "Por sitio y merge",
        "sub_pool": "Pool general de tickets",
        "sub_site": "Por sitio (sin /sites) con IA y merge",
        "pool_size": "Tamaño del pool (total)",
        "page_size": "Tamaño por página API (<=100)",
        "start_index": "Start index",
        "debug": "Debug request",
        "load_pool": "Cargar pool",
        "last_request": "Última solicitud (debug)",
        "env_error": "No pude leer .env: {error}",
        "env_info": "Crea un .env basado en .env.example y vuelve a ejecutar.",
        "progress_start": "Iniciando...",
        "progress_page": "Cargando página {page}/{total}...",
        "progress_done": "Listo. Tickets cargados: {count}",
        "need_pool": "Primero carga un pool en la pestaña 'Pool general'.",
        "no_sites": "No pude derivar sitios desde el pool (campo 'site' vacío).",
        "select_site": "Selecciona un sitio",
        "max_show": "Máximo a mostrar en tabla",
        "refresh_parent": "Refrescar padre desde API al merge",
        "only_open": "Mostrar solo Open y Unassigned (recomendado)",
        "load_site": "Cargar tickets de este sitio",
        "site_tickets": "Tickets del sitio",
        "manual_merge": "Merge manual",
        "select_parent": "Selecciona el ticket padre",
        "select_children": "Selecciona los tickets hijos",
        "apply_merge": "Aplicar merge",
        "merge_done": "Merge completado. Padre={parent}, hijos={children}",
        "merge_failed": "No se pudo completar el merge. Detalle: {error}",
        "need_two": "Se necesitan al menos 2 tickets para hacer un merge.",
        "ai_groups": "IA: agrupar tickets similares (posibles duplicados)",
        "ai_button": "Generar grupos con IA",
        "ai_fail": "No se pudo agrupar con IA. Detalle: {error}",
        "groups_count": "Grupos sugeridos: {count}",
        "evidence": "Evidencia: ",
        "empty_group": "Grupo vacío: IDs no están en la tabla actual (quizá ya fueron mergeados).",
        "parent_pick": "Elegir padre (ticket principal)",
        "children_pick": "Selecciona hijos a mergear en este padre",
        "merge_group": "Aplicar merge a este grupo",
        "needs_review": "Requiere revisión",
        "confidence": "confianza={value}",
        "lang_label": "Idioma",
        "pool_loaded": "Tickets en '{site}': {total} (mostrando {shown})",
        "need_parent_child": "Necesitas un padre y al menos 1 hijo.",
        "merge_busy": "Hay un merge en curso. Espera a que termine para continuar.",
    },
    "en": {
        "app_title": "SDP Auto Merge",
        "tab_pool": "General pool",
        "tab_site": "By site and merge",
        "sub_pool": "General ticket pool",
        "sub_site": "By site (no /sites) with AI and merge",
        "pool_size": "Pool size (total)",
        "page_size": "API page size (<=100)",
        "start_index": "Start index",
        "debug": "Debug request",
        "load_pool": "Load pool",
        "last_request": "Last request (debug)",
        "env_error": "Could not read .env: {error}",
        "env_info": "Create a .env based on .env.example and try again.",
        "progress_start": "Starting...",
        "progress_page": "Loading page {page}/{total}...",
        "progress_done": "Done. Tickets loaded: {count}",
        "need_pool": "Load a pool first in the 'General pool' tab.",
        "no_sites": "Could not derive sites from the pool (empty 'site' field).",
        "select_site": "Select a site",
        "max_show": "Max to show in table",
        "refresh_parent": "Refresh parent from API after merge",
        "only_open": "Show only Open and Unassigned (recommended)",
        "load_site": "Load tickets for this site",
        "site_tickets": "Site tickets",
        "manual_merge": "Manual merge",
        "select_parent": "Select parent ticket",
        "select_children": "Select child tickets",
        "apply_merge": "Apply merge",
        "merge_done": "Merge completed. Parent={parent}, children={children}",
        "merge_failed": "Merge failed. Detail: {error}",
        "need_two": "At least 2 tickets are required to merge.",
        "ai_groups": "AI: group similar tickets (possible duplicates)",
        "ai_button": "Generate groups with AI",
        "ai_fail": "AI grouping failed. Detail: {error}",
        "groups_count": "Suggested groups: {count}",
        "evidence": "Evidence: ",
        "empty_group": "Empty group: IDs are not in the current table (maybe already merged).",
        "parent_pick": "Choose parent (main ticket)",
        "children_pick": "Select children to merge into this parent",
        "merge_group": "Apply merge to this group",
        "needs_review": "Needs review",
        "confidence": "confidence={value}",
        "lang_label": "Language",
        "pool_loaded": "Tickets in '{site}': {total} (showing {shown})",
        "need_parent_child": "You need a parent and at least 1 child.",
        "merge_busy": "A merge is in progress. Please wait until it finishes.",
    },
}


st.set_page_config(page_title="SDP Auto Merge", layout="wide")

lang = st.sidebar.selectbox("Idioma / Language", options=["es", "en"], index=0, key="ui_language")


def t(key: str) -> str:
    return LANG.get(lang, LANG["es"]).get(key, key)


st.title(t("app_title"))

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
    st.error(t("env_error").format(error=e))
    st.info(t("env_info"))
    st.stop()

tab_pool, tab_site = st.tabs([t("tab_pool"), t("tab_site")])

# ---------------- TAB 1 ----------------
with tab_pool:
    st.subheader(t("sub_pool"))

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        pool_size = st.number_input(t("pool_size"), min_value=1, max_value=5000, value=500, step=50)
    with c2:
        page_size = st.number_input(t("page_size"), min_value=1, max_value=100, value=100, step=10)
    with c3:
        start_index = st.number_input(t("start_index"), min_value=1, value=1, step=1)
    with c4:
        debug = st.checkbox(t("debug"), value=False)

    if st.button(t("load_pool"), type="primary"):
        try:
            st.session_state["pool_data"] = fetch_pool(client, int(pool_size), int(page_size), int(start_index))
            st.session_state["site_source_pool"] = st.session_state["pool_data"]
            st.session_state["site_tickets"] = []
            st.session_state["ai_groups"] = []
        except (ServiceDeskApiError, ValueError) as e:
            st.error(str(e))

    if debug and client.last_http:
        st.caption(t("last_request"))
        st.code(
            f"METHOD: {client.last_http.method}\nURL: {client.last_http.url}\nPARAMS: {client.last_http.params}",
            language="text",
        )

    data = st.session_state.get("pool_data", [])
    if data:
        st.dataframe(data, use_container_width=True, height=520)

# ---------------- TAB 2 ----------------
with tab_site:
    clear_merge_lock_if_stale()
    st.subheader(t("sub_site"))

    src_pool = st.session_state.get("site_source_pool", [])
    if not src_pool:
        st.warning(t("need_pool"))
        st.stop()

    sites = derive_sites_from_tickets(src_pool)
    if not sites:
        st.warning(t("no_sites"))
        st.stop()

    selected_site = st.selectbox(t("select_site"), options=sites)

    # Reset manual plan si cambia el sitio
    if st.session_state.get("manual_site") != selected_site:
        st.session_state["manual_site"] = selected_site
        st.session_state.pop("manual_parent_single", None)
        st.session_state.pop("manual_children_single", None)
        st.session_state["ai_groups"] = []

    cA, cB, cC = st.columns([1, 1, 1])
    with cA:
        max_to_show = st.number_input(t("max_show"), min_value=50, max_value=10000, value=2000, step=50)
    with cB:
        refresh_after_merge = st.checkbox(t("refresh_parent"), value=True)
    with cC:
        show_only_open_unassigned = st.checkbox(t("only_open"), value=True)

    if st.button(t("load_site"), type="primary"):
        site_all = [t for t in src_pool if t.get("site") == selected_site]
        if show_only_open_unassigned:
            site_all = [t for t in site_all if (t.get("status") or "").strip().lower() == "open" and (t.get("technician") or "").strip().lower() in ("", "unassigned", "none", "-", "null")]
        st.session_state["site_tickets"] = site_all[: int(max_to_show)]
        st.success(
            t("pool_loaded").format(
                site=selected_site,
                total=len(site_all),
                shown=len(st.session_state["site_tickets"]),
            )
        )

    site_tickets = st.session_state.get("site_tickets", [])
    if not site_tickets:
        st.stop()

    st.markdown(f"### {t('site_tickets')}")
    st.dataframe(site_tickets, use_container_width=True, height=420)

    st.divider()

    # Tablero manual siempre visible (sin necesidad de IA)
    st.markdown(f"## {t('manual_merge')}")
    by_id = {str(t.get("id")): t for t in site_tickets}
    compiled = sorted(site_tickets, key=lambda t: _to_int_id(t.get("id", "")))
    st.dataframe(compiled, use_container_width=True, height=300)

    all_ids = [str(t.get("id")) for t in compiled if str(t.get("id", "")).strip()]
    if len(all_ids) >= 2:
        suggested_parent = build_default_best_parent(all_ids, by_id)
        default_parent = suggested_parent if suggested_parent in all_ids else all_ids[0]

        parent_id = st.selectbox(
            t("select_parent"),
            options=all_ids,
            index=all_ids.index(default_parent),
            format_func=lambda tid: ticket_label(by_id, tid),
            key="manual_parent_single",
        )

        child_options = [x for x in all_ids if x != parent_id]
        child_ids = st.multiselect(
            t("select_children"),
            options=child_options,
            format_func=lambda tid: ticket_label(by_id, tid),
            key="manual_children_single",
        )

        merge_busy = bool(st.session_state.get("merge_in_progress"))
        apply_merge = st.button(
            t("apply_merge"),
            type="primary",
            disabled=merge_busy or (not bool(child_ids)),
        )
        if apply_merge:
            if st.session_state.get("merge_in_progress"):
                st.info(t("merge_busy"))
                st.stop()
            start_merge_lock()
            try:
                _res = client.merge_requests(parent_id=parent_id, child_ids=child_ids)
                st.success(t("merge_done").format(parent=parent_id, children=child_ids))
                apply_merge_local_update(client, selected_site, parent_id, child_ids, refresh_parent=refresh_after_merge)
                update_ai_groups_after_merge(parent_id, child_ids)
            except ServiceDeskApiError as e:
                st.error(t("merge_failed").format(error=e))
            finally:
                st.session_state["merge_in_progress"] = False
                st.session_state["merge_started_at"] = None
                st.rerun()
    else:
        st.info(t("need_two"))

    st.divider()
    st.markdown(f"## {t('ai_groups')}")

    if st.button(t("ai_button")):
        try:
            # Puedes pasar todos los tickets mostrados; la IA prioriza elegibles, y tu app valida
            ai = group_tickets_with_ai(site_tickets, site_name=selected_site)
            groups = ai.get("groups", []) or []

            # Normaliza padre con jerarquía local
            groups = normalize_groups_choose_parent(groups, site_tickets)

            # Post-validación anti-alucinación (SM/AP vs Cam/NVR/DA)
            groups = post_validate_ai_groups(groups, site_tickets)

            # Priorizar jerarquías (SM y switches) antes que duplicados
            hierarchy_groups = []
            hierarchy_groups.extend(groups_from_sm_camera_reference(site_tickets, groups))
            hierarchy_groups.extend(groups_from_switch_camera_reference(site_tickets, groups))

            used: set[str] = set()
            hierarchy_groups = _filter_groups_without_overlap(hierarchy_groups, used)
            groups = _filter_groups_without_overlap(groups, used)

            groups = hierarchy_groups + groups

            # Si aún quedó vacío, fallback sin IA (para no mostrar "nada")
            real_groups = [g for g in groups if isinstance(g, dict) and str(g.get("reason", "")) != "ALL_TICKETS_NO_MERGE"]
            if not real_groups:
                fb = fallback_groups_no_ai(site_tickets)
                used_fb: set[str] = set()
                fb = _filter_groups_without_overlap(fb, used_fb)
                groups = fb

            # Asegura tablero compilado al final
            groups = ensure_compiled_group(groups, site_tickets)

            st.session_state["ai_groups"] = groups

        except AIError as e:
            st.error(t("ai_fail").format(error=e))

    groups = st.session_state.get("ai_groups")
    if not groups:
        st.stop()

    visible_groups = [
        g
        for g in (groups or [])
        if isinstance(g, dict) and str(g.get("reason", "")) != "ALL_TICKETS_NO_MERGE"
    ]
    st.success(t("groups_count").format(count=len(visible_groups)))

    # Render de grupos IA
    for i, g in enumerate(visible_groups, start=1):
        if not isinstance(g, dict):
            continue

        reason = str(g.get("reason", "") or "").strip()
        if reason == "ALL_TICKETS_NO_MERGE":
            continue

        parent_id = str(g.get("parent_id", "") or "").strip()
        child_ids = [str(x).strip() for x in (g.get("child_ids") or []) if str(x).strip()]

        ids = []
        if parent_id:
            ids.append(parent_id)
        ids.extend([x for x in child_ids if x and x != parent_id])
        ids = [x for x in ids if x in by_id]

        conf = g.get("confidence", None)
        needs_review = bool(g.get("needs_review", False))
        evidence = g.get("evidence", []) or []

        reason_label = format_reason(reason, lang)
        title_bits = [f"Grupo #{i}", f"{len(ids)} tickets", reason_label]
        if conf is not None:
            try:
                title_bits.append(t("confidence").format(value=float(conf)))
            except Exception:
                pass
        if needs_review:
            title_bits.append(t("needs_review"))

        with st.expander(" - ".join(title_bits), expanded=False):
            if evidence:
                st.caption(t("evidence") + " | ".join([str(x) for x in evidence][:12]))

            rows = [by_id[x] for x in ids]
            st.dataframe(rows, use_container_width=True, height=220)

            if not ids:
                st.warning(t("empty_group"))
                continue

            is_done = bool(g.get("merged_done")) or ("MERGED_DONE" in reason)

            default_parent = parent_id if parent_id in ids else ids[0]
            new_parent = st.selectbox(
                t("parent_pick"),
                options=ids,
                index=ids.index(default_parent),
                format_func=lambda tid: ticket_label(by_id, tid),
                key=f"parent_sel_{i}",
                disabled=is_done,
            )

            selectable_children = [x for x in ids if x != new_parent]
            selected_children = st.multiselect(
                t("children_pick"),
                options=selectable_children,
                default=selectable_children,
                format_func=lambda tid: ticket_label(by_id, tid),
                key=f"children_sel_{i}",
                disabled=is_done,
            )

            merge_busy = bool(st.session_state.get("merge_in_progress"))
            do_merge = st.button(
                t("merge_group"),
                type="primary",
                key=f"merge_btn_{i}",
                disabled=is_done or merge_busy,
            )

            if do_merge:
                if not new_parent or not selected_children:
                    st.warning(t("need_parent_child"))
                else:
                    if st.session_state.get("merge_in_progress"):
                        st.info(t("merge_busy"))
                        st.stop()
                    start_merge_lock()
                    try:
                        _res = client.merge_requests(parent_id=new_parent, child_ids=selected_children)
                        st.success(t("merge_done").format(parent=new_parent, children=selected_children))

                        apply_merge_local_update(
                            client,
                            selected_site,
                            new_parent,
                            selected_children,
                            refresh_parent=refresh_after_merge,
                        )

                        update_ai_groups_after_merge(new_parent, selected_children)
                        g["merged_done"] = True
                    except ServiceDeskApiError as e:
                        st.error(t("merge_failed").format(error=e))
                    finally:
                        st.session_state["merge_in_progress"] = False
                        st.session_state["merge_started_at"] = None
                        st.rerun()
