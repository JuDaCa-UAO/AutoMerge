from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

from openai import OpenAI


class AIError(RuntimeError):
    pass


IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


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
    model = (os.getenv("OPENAI_MODEL") or "gpt-4.1-mini").strip()

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
        compact.append(
            {
                "id": tid,
                "subject": subject,
                "site": site,
                "created": str(t.get("created", "")).strip(),
                "status": str(t.get("status", "")).strip(),
                "requester": str(t.get("requester", "")).strip(),
                "technician": str(t.get("technician", "")).strip(),
                "ips": IP_RE.findall(subject),
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

ANTI-ALUCINACIÓN SM vs CÁMARA:
- PROHIBIDO agrupar SM/AP con cámara/NVR/DA solo por compartir un número como "11", "21", "31".
- SOLO puedes agrupar SM/AP con cámara/NVR/DA si existe evidencia fuerte:
  A) comparten una IPv4 COMPLETA idéntica en subject, o
  B) el subject del SM/AP contiene explícitamente referencias a cámaras ("Cam 10-12", etc.)

PATRONES IP (contexto):
- AP: 192.168.0.xx (xx típicamente decenas: 10,20,30,...)
- SM asociado suele ser AP+1: 0.10 -> 0.11; 0.20 -> 0.21
- NVR: 192.168.1.103
- Router: 192.168.1.254
- DA: 192.168.1.20x (200,201,203,...)

AGRUPACIÓN (orden):
1) Duplicados exactos (subject igual normalizando espacios y may/min)
2) Misma IPv4 completa
3) Similaridad fuerte de subject (mismo tipo de dispositivo). Si es probable pero no seguro => needs_review=true

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

TICKETS:
{compact}
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
