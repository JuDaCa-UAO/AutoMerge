from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from openai import OpenAI


class AIError(RuntimeError):
    pass


def group_tickets_with_ai(tickets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Input tickets: [{id, subject, site, created, status, requester, technician}, ...]
    Output:
      {"groups":[{"reason":"...", "parent_id":"123", "child_ids":["124","125"]}, ...]}
    """
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    model = (os.getenv("OPENAI_MODEL") or "gpt-4.1-mini").strip()

    if not api_key:
        raise AIError("Falta OPENAI_API_KEY en tu .env")

    client = OpenAI(api_key=api_key)

    compact = [
        {
            "id": str(t.get("id", "")).strip(),
            "subject": str(t.get("subject", "")).strip(),
            "site": str(t.get("site", "")).strip(),
            "created": str(t.get("created", "")).strip(),
            "status": str(t.get("status", "")).strip(),
            "requester": str(t.get("requester", "")).strip(),
            "technician": str(t.get("technician", "")).strip(),
        }
        for t in tickets
        if t.get("id") and t.get("subject") and t.get("site")
    ]

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

    # ✅ Prompt enriquecido con reglas de IP (AP/SM/NVR/Router/DA)
    prompt = f"""
Eres un analista NOC senior. Tu tarea es proponer grupos de tickets para MERGE en ServiceDesk (ManageEngine SDP).

FILTRO ESTRICTO (NO lo rompas):
- Usa ÚNICAMENTE tickets con:
  1) status == "Open" (estricto)
  2) technician SIN asignar ("", "Unassigned", "None", null, "-")
  3) requester == "ServiceDesk" (PRIORIDAD ALTA). Si no hay suficientes, puedes usar otros requester SOLO si son duplicados clarísimos.
- NO mezcles sitios.
- NO inventes IDs.
- Un ticket no debe aparecer en más de un grupo (EXCEPTO el bloque final ALL_TICKETS_NO_MERGE).

OBJETIVO:
Agrupar tickets que representen el mismo incidente/intermitencia para poder mergearlos.

Estrategia de agrupación (en orden):
1) DUPLICADOS EXACTOS:
   - Mismo sitio
   - subject exactamente igual (normaliza espacios y mayúsculas/minúsculas, pero NO cambies números/IPs)

2) SIMILITUD FUERTE:
   - Misma IP (IPv4) en subject
   - Subject casi igual (Down/Up, ping, timestamps, IDs internos, etc.)
   - Familias de cámaras (.1.1–.1.x o .2.1–.2.x)
   - Keywords: camera/cam, ap, switch, router, radio, nvr, da, digital acoustics, speaker

3) INFERENCIA POR PATRONES DE IP (IMPORTANTE):
En algunos sitios los dispositivos NO aparecen con nombre "SM"/"AP" sino por IP.
Usa estas reglas para inferir el tipo de dispositivo incluso si el subject solo trae IP:

   A) APs:
      - Los AP suelen estar en 192.168.0.xx donde xx son decenas típicas, ej:
        192.168.0.10, 192.168.0.20, 192.168.0.30, ...
      - (En general: AP ~ último octeto termina en 0 dentro de 10-99, o coincide con patrón de “decenas”).

   B) SMs:
      - Un SM puede corresponder a uno o más AP.
      - Si existe un AP con IP 192.168.0.10, su SM correspondiente suele ser 192.168.0.11.
      - El siguiente “bloque” SM asociado al siguiente AP base 192.168.0.20 suele ser 192.168.0.21.
      - Regla práctica:
        - AP base: 192.168.0.(N0)
        - SM asociado: 192.168.0.(N0+1)
      - Por tanto, si ves pares tipo (0.10 y 0.11), (0.20 y 0.21), etc., es muy probable que estén relacionados y deban agruparse.

   C) NVR:
      - Todos los NVR tienen IP 192.168.1.103 (si ves esa IP, clasifícalo como NVR).

   D) Routers:
      - Router principal típicamente 192.168.1.254 (si ves esa IP, clasifícalo como Router).

   E) DA (Digital Acoustics):
      - Dispositivos DA (para transmitir voz a speakers) usan IP 192.168.1.20x
        Ejemplos: 192.168.1.200, 192.168.1.201, 192.168.1.203 (saltan de 1 en 1, no necesariamente continuos).
      - Si ves 192.168.1.20x, clasifícalo como DA.

JERARQUÍA PARA DECIDIR PADRE (para merges dentro del grupo):
Router -> Switch 100/Core -> Switch secundario -> Radios -> Switch terciario -> Endpoints (AP/SM/Cámaras/NVR/DA)

REGLA ESPECIAL Switch 100/Core:
- Un ticket que incluya IP con último octeto 100-109 (patrón ".10x") representa Switch 100/Core.
  Ej: ".10x SM" debe tener prioridad como PADRE sobre "SM" genérico.

SELECCIÓN DEL PADRE:
- parent_id debe ser el ticket MÁS ARRIBA en la jerarquía dentro del grupo.
- Si hay empate en jerarquía, el PADRE es el ticket más viejo usando esta regla:
  -> ID numéricamente menor = ticket más viejo.

SALIDA:
Devuelve SOLO JSON válido con:
{{
  "groups": [
    {{
      "reason": "texto corto",
      "parent_id": "ID",
      "child_ids": ["ID","ID"]
    }}
  ]
}}

REGLA EXTRA:
- SIEMPRE agrega al final:
  reason = "ALL_TICKETS_NO_MERGE"
  parent_id = el ID numéricamente menor de TODOS los tickets considerados
  child_ids = []

TICKETS:
{compact}
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
        raise AIError(f"No pude parsear la respuesta JSON: {e}\nRaw:\n{resp.output_text}") from e
