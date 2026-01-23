from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from openai import OpenAI


class AIError(RuntimeError):
    pass


def refine_candidate_groups_with_ai(
    *,
    site_name: str,
    candidate_groups: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    candidate_groups:
      [
        {
          "reason_hint": "EXACT_SUBJECT" | "DEVICE_BUCKET_FUZZY" | ...,
          "tickets": [{id, subject, created, status, requester, technician}, ...]
        }, ...
      ]

    output:
      {"groups":[{"reason":"...", "parent_id":"123", "child_ids":["124","125"]}, ...]}
    """
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    model = (os.getenv("OPENAI_MODEL") or "gpt-4.1-mini").strip()

    if not api_key:
        raise AIError("Falta OPENAI_API_KEY en tu .env")

    client = OpenAI(api_key=api_key)

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
                    },
                    "required": ["reason", "parent_id", "child_ids"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["groups"],
        "additionalProperties": False,
    }

    # ⚠️ Reglas: estrictas para NO mezclar SMxx con CAMxx si no hay evidencia
    prompt = f"""
Eres un analista NOC senior. Tu tarea: tomar grupos CANDIDATOS (pre-agrupados heurísticamente) y:
1) validar si realmente son mergeables
2) escoger parent_id y child_ids según jerarquía
3) NO inventar IDs ni mezclar sitios.

SITIO ACTUAL: {site_name}
NO MEZCLES SITIOS.

ELEGIBILIDAD (ESTRICTO, NO NEGOCIABLE):
- Solo considera tickets con:
  - status == "Open"
  - technician SIN asignar (Unassigned/"")
  - requester == "ServiceDesk"
- Si un ticket no cumple, NO PUEDE estar en ningún grupo final.
- Si por alguna razón hay mezcla, simplemente omite ese ticket.

ANTI-CONFUSIÓN (CRÍTICO):
- NO agrupes SM 11 con Cam 11 solo por el número.
- Solo relaciona SM con cámaras si el subject del SM incluye explícitamente cámaras (ej "CAM 13-15", "Cam 17A-B", etc).
- No uses coincidencia de último octeto ni número suelto para inferir relaciones entre tipos diferentes.

CONTEXTO IP (para interpretar subjects por IP):
- Routers: 192.168.1.254
- NVR: 192.168.1.103
- DA: 192.168.1.20x (ej 192.168.1.200, .201, .203)
- AP: 192.168.0.xx donde xx típicamente son decenas (10,20,30,...)
- SM asociado a AP suele ser AP+1 (10->11, 20->21, etc) PERO SOLO si la IP completa aparece en subject.

JERARQUÍA PARA PADRE (IMPORTANTE: ESTA ES TU REGLA NUEVA):
AP -> SM -> Switch secundario -> Radios -> Switch terciario -> Endpoints (cámaras, NVR, DA)
- Si hay empate, parent_id = ID numéricamente menor (más viejo).

REGLA EXTRA:
- Priorización de requester: si hay varios tickets equivalentes, prefiere los de requester "ServiceDesk" (aunque igual es requisito estricto).
- Grupos deben tener 2+ tickets.
- Un ticket NO puede estar en más de un grupo.

SALIDA: SOLO JSON válido con schema:
{{
  "groups": [
    {{
      "reason": "texto corto",
      "parent_id": "ID",
      "child_ids": ["ID","ID"]
    }}
  ]
}}

CANDIDATOS (cada grupo trae tickets completos):
{json.dumps(candidate_groups, ensure_ascii=False)}
""".strip()

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
        raise AIError(f"No pude parsear JSON: {e}\nRaw:\n{resp.output_text}") from e
