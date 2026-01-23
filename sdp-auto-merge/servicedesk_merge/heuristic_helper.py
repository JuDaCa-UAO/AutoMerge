from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
SM_RE = re.compile(r"\b(?:SIG-)?SM\s*([0-9]{1,3})\b", re.IGNORECASE)
AP_RE = re.compile(r"\b(?:SIG-)?AP\s*([0-9]{1,3})\b", re.IGNORECASE)
CAM_RE = re.compile(r"\bCAM(?:ERA)?\s*([0-9]{1,3})(?:\s*[-–]\s*([0-9]{1,3}))?([A-Z]?)\b", re.IGNORECASE)

# “.10x” interpretado como core/switch100 -> último octeto 100-109
def _last_octets(subject: str) -> List[int]:
    out: List[int] = []
    for ip in IP_RE.findall(subject or ""):
        parts = ip.split(".")
        if len(parts) == 4:
            try:
                out.append(int(parts[3]))
            except Exception:
                pass
    return out


def _clean_spaces(s: str) -> str:
    return " ".join((s or "").strip().split())


def normalize_subject(subject: str) -> str:
    """
    Normalización “segura”:
    - lower
    - colapsa espacios
    - NO cambia números/IPs
    - quita ruido común (down/up/ping) sin tocar tokens importantes
    """
    s = (subject or "").lower()
    s = _clean_spaces(s)

    # quita timestamps muy típicos si aparecen como ruido
    s = re.sub(r"\b\d{1,2}:\d{2}\b", " ", s)
    s = re.sub(r"\b(am|pm)\b", " ", s)
    s = _clean_spaces(s)

    # reduce variantes
    s = s.replace("is down", "down").replace("device down", "down")
    s = s.replace("is up", "up")
    s = _clean_spaces(s)
    return s


def requester_is_servicedesk(req: str) -> bool:
    return (req or "").strip().lower() in ("servicedesk", "service desk")


def technician_is_unassigned(tech: str) -> bool:
    t = (tech or "").strip().lower()
    return t in ("", "unassigned", "none", "-", "null")


def status_is_open(status: str) -> bool:
    return (status or "").strip().lower() == "open"


def ip_kind(ip: str) -> str:
    """
    Clasificación por IP exacta.
    """
    if ip == "192.168.1.254":
        return "router"
    if ip == "192.168.1.103":
        return "nvr"
    if ip.startswith("192.168.1.20"):
        return "da"
    if ip.startswith("192.168.0."):
        return "ap_or_sm"
    return "ip"


def infer_ap_sm_from_ip(ip: str) -> Tuple[str, str]:
    """
    AP suele ser 192.168.0.(decenas) -> 10,20,30...
    SM asociado suele ser AP+1: 10->11, 20->21, etc.
    Retorna (kind, key)
    """
    try:
        last = int(ip.split(".")[-1])
    except Exception:
        return ("ip", ip)

    # AP base: 10,20,30... (último dígito 0 y >=10)
    if last >= 10 and last % 10 == 0:
        return ("ap", f"AP_IP_{ip}")

    # SM asociado: 11,21,31... (último dígito 1 y >=11)
    if last >= 11 and last % 10 == 1:
        return ("sm", f"SM_IP_{ip}")

    return ("ip", ip)


def device_type_and_key(subject: str) -> Tuple[str, str]:
    """
    MUY IMPORTANTE: no mezclar SM11 con CAM11.
    - SM solo si hay token SM explícito (SM / SIG-SM)
    - CAM solo si hay token CAM explícito
    - no usar “número suelto” para inferir nada
    """
    s = subject or ""
    s_norm = s.lower()

    # IPs especiales primero
    ips = IP_RE.findall(s_norm)
    if ips:
        for ip in ips:
            k = ip_kind(ip)
            if k in ("router", "nvr", "da"):
                return (k, ip)
        # switch100 por octeto 100-109 (si hay IPs)
        if any(100 <= o <= 109 for o in _last_octets(s_norm)):
            return ("switch100", "SWITCH100")
        # AP/SM por rango 192.168.0.x
        for ip in ips:
            if ip.startswith("192.168.0."):
                kind, key = infer_ap_sm_from_ip(ip)
                if kind in ("ap", "sm"):
                    return (kind, key)
        # si no, usar primera IP como key
        return ("ip", ips[0])

    # Tokens explícitos
    m_sm = SM_RE.search(s_norm)
    if m_sm:
        num = m_sm.group(1)
        return ("sm", f"SM_{num}")

    m_ap = AP_RE.search(s_norm)
    if m_ap:
        num = m_ap.group(1)
        return ("ap", f"AP_{num}")

    m_cam = CAM_RE.search(s_norm)
    if m_cam:
        a = m_cam.group(1)
        b = m_cam.group(2) or ""
        suf = m_cam.group(3) or ""
        if b:
            return ("camera", f"CAM_{a}-{b}{suf}")
        return ("camera", f"CAM_{a}{suf}")

    # palabras clave genéricas
    if "router" in s_norm or "gateway" in s_norm or "gw" in s_norm:
        return ("router", "ROUTER")
    if "switch" in s_norm:
        # no asumir SM
        return ("switch", "SWITCH")
    if "radio" in s_norm:
        return ("radio", "RADIO")

    return ("unknown", "UNKNOWN")


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


@dataclass
class CandidateGroup:
    reason: str
    ids: List[str]


def build_candidate_groups(
    tickets: List[Dict],
    *,
    fuzzy_threshold: float = 0.90,
    min_group_size: int = 2,
    max_candidates: int = 60,
) -> List[CandidateGroup]:
    """
    Helper heurístico:
    - agrupa rápido sin IA
    - SOLO propone candidatos; la IA decide jerarquía/padre (en la siguiente etapa)
    """
    # Index tickets
    by_id: Dict[str, Dict] = {str(t.get("id")): t for t in tickets if t.get("id")}

    # 1) exact subject duplicates (normalizando espacios/case)
    subject_map: Dict[str, List[str]] = {}
    for tid, t in by_id.items():
        subj = normalize_subject(str(t.get("subject", "")))
        subject_map.setdefault(subj, []).append(tid)

    groups: List[CandidateGroup] = []
    used: set[str] = set()

    for subj_norm, ids in subject_map.items():
        if len(ids) >= min_group_size:
            ids_sorted = sorted(ids, key=lambda x: int(x))
            # no usamos usados aquí todavía (permitimos overlap en candidatos,
            # la IA luego decide y el UI impide merge sin revisión)
            groups.append(CandidateGroup(reason="EXACT_SUBJECT", ids=ids_sorted))

    # 2) same device key bucket + fuzzy within bucket
    buckets: Dict[Tuple[str, str], List[str]] = {}
    subj_norm_map: Dict[str, str] = {}
    for tid, t in by_id.items():
        subj = str(t.get("subject", ""))
        subj_n = normalize_subject(subj)
        subj_norm_map[tid] = subj_n
        dtype, dkey = device_type_and_key(subj)
        buckets.setdefault((dtype, dkey), []).append(tid)

    # Fuzzy clustering per bucket (simple union-find)
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for (dtype, dkey), ids in buckets.items():
        if len(ids) < 2:
            continue

        # anti-confusión: solo comparar dentro del mismo tipo (ya lo estamos haciendo),
        # y además: SM vs camera ya está separado por device_type_and_key.
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                sa, sb = subj_norm_map[a], subj_norm_map[b]

                # señales fuertes: misma IP completa
                ips_a = set(IP_RE.findall(sa))
                ips_b = set(IP_RE.findall(sb))
                if ips_a and ips_b and (ips_a & ips_b):
                    union(a, b)
                    continue

                # fuzzy
                if similarity(sa, sb) >= fuzzy_threshold:
                    union(a, b)

    clusters: Dict[str, List[str]] = {}
    for tid in by_id.keys():
        r = find(tid)
        clusters.setdefault(r, []).append(tid)

    for root, ids in clusters.items():
        if len(ids) >= min_group_size:
            ids_sorted = sorted(ids, key=lambda x: int(x))
            groups.append(CandidateGroup(reason="DEVICE_BUCKET_FUZZY", ids=ids_sorted))

    # 3) limitar candidatos (para no saturar UI / IA)
    # Orden: primero exact, luego fuzzy
    def score(g: CandidateGroup) -> Tuple[int, int]:
        pri = 0 if g.reason == "EXACT_SUBJECT" else 1
        return (pri, -len(g.ids))

    groups = sorted(groups, key=score)
    # dedupe por set de ids
    seen_sets: set[Tuple[str, ...]] = set()
    final: List[CandidateGroup] = []
    for g in groups:
        key = tuple(g.ids)
        if key in seen_sets:
            continue
        seen_sets.add(key)
        final.append(g)
        if len(final) >= max_candidates:
            break

    return final
