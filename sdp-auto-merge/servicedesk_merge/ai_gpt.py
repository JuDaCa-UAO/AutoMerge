from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

from openai import OpenAI


class AIError(RuntimeError):
    pass


IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def _normalize_subject(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _eligible_for_merge(t: Dict[str, Any]) -> bool:
    status = (t.get("status") or "").strip().lower()
    technician = (t.get("technician") or "").strip().lower()
    requester = (t.get("requester") or "").strip()
    is_open = status == "open"
    tech_unassigned = technician in ("", "unassigned", "none", "-", "null")
    req_ok = requester == "ServiceDesk"
    return is_open and tech_unassigned and req_ok


def _infer_device_hint(subject: str) -> str:
    subj = (subject or "").lower()
    ips = IP_RE.findall(subj)

    if "192.168.1.254" in ips or "router" in subj or "gateway" in subj or " gw" in subj:
        return "router"
    if "192.168.1.103" in ips or "nvr" in subj:
        return "nvr"
    if any(ip.startswith("192.168.1.20") for ip in ips) or "digital acoustics" in subj or " da" in subj:
        return "da"
    if "camera" in subj or re.search(r"\bcam\b", subj):
        return "camera"
    if "radio" in subj:
        return "radio"
    if any(ip.startswith("192.168.0.") for ip in ips):
        parts = [p.split(".")[-1] for p in ips if p.startswith("192.168.0.")]
        for last in parts:
            try:
                o = int(last)
            except Exception:
                continue
            if 10 <= o <= 99:
                if o % 10 == 0:
                    return "ap"
                if o % 10 == 1:
                    return "sm"
    if re.search(r"(^|\\s)ap(\\s|$)", subj):
        return "ap"
    if re.search(r"(^|\\s)sm(\\s|$)", subj):
        return "sm"
    if "switch" in subj or re.search(r"\\bsw\\b", subj):
        return "switch"
    return "unknown"


def _build_candidate_groups(compact: List[Dict[str, Any]]) -> Dict[str, List[List[str]]]:
    by_subject: Dict[str, List[str]] = {}
    by_ip: Dict[str, List[str]] = {}
    for t in compact:
        tid = t["id"]
        subj_norm = t.get("subject_norm", "")
        if subj_norm:
            by_subject.setdefault(subj_norm, []).append(tid)
        for ip in t.get("ips", []) or []:
            by_ip.setdefault(ip, []).append(tid)

    exact_subject_groups = [ids for ids in by_subject.values() if len(ids) >= 2]
    same_ip_groups = [ids for ids in by_ip.values() if len(ids) >= 2]

    # Cap size to avoid oversized prompts
    exact_subject_groups = exact_subject_groups[:200]
    same_ip_groups = same_ip_groups[:200]

    return {
        "exact_subject_groups": exact_subject_groups,
        "same_ip_groups": same_ip_groups,
    }


def group_tickets_with_ai(
    tickets: List[Dict[str, Any]],
    *,
    site_name: str = "",
) -> Dict[str, Any]:
    """
    Devuelve:
      {"groups":[{"reason","parent_id","child_ids","confidence","needs_review","evidence"}]}
    """
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    model = (os.getenv("OPENAI_MODEL") or "gpt-4.1").strip()

    if not api_key:
        raise AIError("Falta OPENAI_API_KEY en tu .env")

    client = OpenAI(api_key=api_key)

    compact = []
    for t in tickets:
        tid = str(t.get("id", "")).strip()
        subject = str(t.get("subject", "")).strip()
        site = str(t.get("site", "")).strip()
        if not tid or not subject or not site:
            continue
        subject_norm = _normalize_subject(subject)
        ips = IP_RE.findall(subject)
        compact.append(
            {
                "id": tid,
                "subject": subject,
                "subject_norm": subject_norm,
                "site": site,
                "created": str(t.get("created", "")).strip(),
                "status": str(t.get("status", "")).strip(),
                "requester": str(t.get("requester", "")).strip(),
                "technician": str(t.get("technician", "")).strip(),
                "ips": ips,
                "eligible": _eligible_for_merge(t),
                "device_hint": _infer_device_hint(subject),
            }
        )

    schema = {
        "type": "object",
        "properties": {
            "groups": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string"},
                        "parent_id": {"type": "string"},
                        "child_ids": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "number"},
                        "needs_review": {"type": "boolean"},
                        "evidence": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["reason", "parent_id", "child_ids", "confidence", "needs_review", "evidence"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["groups"],
        "additionalProperties": False,
    }

    candidate_groups = _build_candidate_groups(compact)

    prompt = f"""
Eres un analista NOC senior. Tu tarea es proponer grupos de tickets para MERGE en ServiceDesk (ManageEngine SDP).

META:
- Queremos que salgan grupos cuando sea razonable.
- Si no estás 100% seguro, igual puedes agrupar pero con needs_review=true y confidence <= 0.65.

REGLAS CRÍTICAS:
- NO mezcles sitios.
- NO inventes IDs.
- Un ticket NO puede aparecer en más de un grupo (excepto ALL_TICKETS_NO_MERGE).

ELEGIBILIDAD (para recomendar merge con confianza):
- Prioriza tickets con:
  1) status == "Open"
  2) technician SIN asignar ("", "Unassigned", "None", null, "-")
  3) requester == "ServiceDesk"
  4) eligible == true

ANTI-ALUCINACIÓN SM vs CÁMARA:
- PROHIBIDO agrupar SM/AP con cámara/NVR/DA solo por compartir un número como "11", "21", "31".
- SOLO puedes agrupar SM/AP con cámara/NVR/DA si existe evidencia fuerte:
  A) comparten una IPv4 COMPLETA idéntica en subject, o
  B) el subject del SM/AP contiene explícitamente referencias a cámaras ("Cam 10-12", etc.)

REGLA PARA SWITCH Y CAMARAS:
- Si un switch menciona camaras en su subject ("Cam 10-12"), puedes agrupar ese switch con esas camaras
  siempre que los tickets de camaras coincidan por numero, rango o IP.

REGLA PARA SM Y CAMARAS:
- Si un SM menciona camaras en su subject ("Cam 10-12"), puedes agrupar ese SM con esas camaras
  siempre que los tickets de camaras coincidan por numero, rango o IP.

PATRONES IP (contexto):
- Cámaras: 192.168.1.x (Puede ser 192.168.2.x pero raro. No consideres como camara las otras ips reservadas que puedan caer dentro de este rango como los NVR o DA)
- AP: 192.168.0.xx (xx típicamente decenas: 10,20,30,...)
- SM asociado suele ser AP+1: 0.10 -> 0.11; 0.20 -> 0.21
- NVR: 192.168.1.103
- Router: 192.168.1.254
- DA: 192.168.1.20x (200,201,203,...)

AGRUPACI?N (orden):
1) Relaciones expl?citas de jerarqu?a (switch/SM con c?maras referenciadas)
2) Duplicados exactos (subject igual normalizando espacios y may/min)
3) Misma IPv4 completa
4) Similaridad fuerte de subject (mismo tipo de dispositivo). Si es probable pero no seguro => needs_review=true
5) Jerarqu?a designada (ver abajo) para elegir PADRE.

PRIORIDAD:
- Si ya agrupaste c?maras con su SM o switch por referencia expl?cita, no vuelvas a agrupar esas c?maras en otros grupos.

SUGERENCIAS CONFIABLES (siempre que no violen reglas):
- exact_subject_groups: agrupa estos tickets, salvo evidencia clara en contra.
- same_ip_groups: agrupa estos tickets, salvo evidencia clara en contra.

JERARQUÍA para sugerir PADRE:
Router -> Switch 100/Core -> Switch secundario -> Radio -> AP -> SM -> Switch terciario -> Endpoints (Cámaras/NVR/DA)

REGLA Switch 100/Core:
- Si aparece IP con último octeto 100-109 (".10x"), suele ser Switch 100/Core.

SELECCIÓN PADRE:
- parent_id = más arriba en jerarquía.
- empate: ID numéricamente menor (más viejo).
- si hay AP/SM/Switch terciario: AP > SM > Switch terciario.

SALIDA:
Solo JSON válido:
{{
  "groups":[
    {{
      "reason":"EXACT_SUBJECT|SAME_IP|SIMILAR_SUBJECT",
      "parent_id":"ID",
      "child_ids":["ID","ID"],
      "confidence":0.0,
      "needs_review":false,
      "evidence":["..."]
    }}
  ]
}}

REGLA EXTRA:
- SIEMPRE agrega al final:
  reason="ALL_TICKETS_NO_MERGE"
  parent_id=ID numéricamente menor del sitio
  child_ids=[]
  confidence=1
  needs_review=false
  evidence=["all_tickets_board"]

SITIO: {site_name}

Cada ticket incluye:
- subject_norm (subject normalizado)
- ips (IPv4 detectadas en subject)
- eligible (cumple reglas de elegibilidad)
- device_hint (tipo probable de dispositivo)

TICKETS:
{compact}

SUGERENCIAS:
{candidate_groups}
""".strip()

    # ✅ Importante: NO pasamos temperature ni max_output_tokens para evitar
    # errores de "Unsupported parameter" dependiendo del modelo.
    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "ticket_groups",
                    "schema": schema,
                    "strict": True,
                }
            },
        )
    except Exception as e:
        raise AIError(f"Error llamando OpenAI: {e}") from e

    try:
        return json.loads(resp.output_text)
    except Exception as e:
        raise AIError(f"No pude parsear la respuesta JSON: {e}\nRaw:\n{resp.output_text}") from e
