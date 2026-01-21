from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests


class ServiceDeskApiError(RuntimeError):
    pass


@dataclass
class HttpDebug:
    method: str
    url: str
    params: Dict[str, Any] | None


class ServiceDeskClient:
    """
    Cliente mínimo SDP v3:
      - GET /requests?input_data=...
      - GET /requests/{id}
      - PUT /requests/{parent}/merge_requests
    """

    def __init__(
        self,
        base_url: str,
        auth_token: str,
        portal_id: str | None = None,
        verify_ssl: bool = True,
        timeout_seconds: int = 30,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.portal_id = portal_id
        self.verify_ssl = verify_ssl
        self.timeout_seconds = timeout_seconds

        self._session = requests.Session()
        self.last_http: HttpDebug | None = None

    def _headers(self, content_type: str | None = None) -> Dict[str, str]:
        h = {
            "Accept": "application/vnd.manageengine.sdp.v3+json, application/json",
            "authtoken": self.auth_token,
        }
        if self.portal_id:
            h["PORTALID"] = self.portal_id
        if content_type:
            h["Content-Type"] = content_type
        return h

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return self.base_url + path

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Dict[str, Any] | None = None,
        data: Dict[str, Any] | None = None,
        json_body: Dict[str, Any] | None = None,
        content_type: str | None = None,
    ) -> Any:
        url = self._url(path)
        self.last_http = HttpDebug(method=method.upper(), url=url, params=params)

        try:
            resp = self._session.request(
                method=method.upper(),
                url=url,
                headers=self._headers(content_type),
                params=params,
                data=data,
                json=json_body,
                timeout=self.timeout_seconds,
                verify=self.verify_ssl,
            )
        except requests.RequestException as e:
            raise ServiceDeskApiError(f"Error de red hacia {url}: {e}") from e

        text = resp.text or ""
        data_json = None
        try:
            data_json = resp.json()
        except Exception:
            data_json = None

        if resp.status_code >= 400:
            detail = ""
            if isinstance(data_json, dict):
                detail = json.dumps(data_json, ensure_ascii=False)[:4000]
            else:
                detail = text[:4000]
            raise ServiceDeskApiError(f"API respondió {resp.status_code} en {method} {url}\nDetalle: {detail}")

        return data_json if data_json is not None else {"raw": text}

    @staticmethod
    def extract_items(payload: Any, key_preference: Optional[str] = None) -> List[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return []
        if key_preference and isinstance(payload.get(key_preference), list):
            return [x for x in payload[key_preference] if isinstance(x, dict)]
        for k in ("requests", "sites"):
            if isinstance(payload.get(k), list):
                return [x for x in payload[k] if isinstance(x, dict)]
        for v in payload.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return [x for x in v if isinstance(x, dict)]
        return []

    @staticmethod
    def has_more(payload: Any, got: int, requested: int) -> bool:
        if isinstance(payload, dict) and isinstance(payload.get("list_info"), dict):
            li = payload["list_info"]
            if "has_more_rows" in li:
                return bool(li["has_more_rows"])
        return got == requested

    def list_requests_page(self, start_index: int, row_count: int) -> Tuple[List[Dict[str, Any]], bool, Any]:
        if row_count > 100:
            raise ValueError("row_count no puede ser > 100 (limitación típica del API).")

        input_data = {"list_info": {"start_index": int(start_index), "row_count": int(row_count)}}
        params = {"input_data": json.dumps(input_data, ensure_ascii=False)}

        payload = self._request("GET", "/requests", params=params)
        items = self.extract_items(payload, key_preference="requests")
        more = self.has_more(payload, got=len(items), requested=row_count)
        return items, more, payload

    def get_request(self, request_id: str) -> Dict[str, Any]:
        payload = self._request("GET", f"/requests/{request_id}")
        if isinstance(payload, dict) and isinstance(payload.get("request"), dict):
            return payload["request"]
        return payload if isinstance(payload, dict) else {"raw": payload}

    def merge_requests(self, parent_id: str, child_ids: List[str]) -> Dict[str, Any]:
        parent_id = str(parent_id)
        children = [str(x) for x in child_ids if str(x) != parent_id]
        payload = {"merge_requests": [{"id": x} for x in children]}

        try:
            return self._request(
                "PUT",
                f"/requests/{parent_id}/merge_requests",
                json_body=payload,
                content_type="application/json",
            )
        except ServiceDeskApiError:
            data = {"input_data": json.dumps(payload, ensure_ascii=False)}
            return self._request(
                "PUT",
                f"/requests/{parent_id}/merge_requests",
                data=data,
                content_type="application/x-www-form-urlencoded",
            )
