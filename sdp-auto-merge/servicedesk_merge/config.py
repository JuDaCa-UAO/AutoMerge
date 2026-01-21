from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


class SettingsError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    api_base_url: str
    auth_token: str
    portal_id: str | None
    verify_ssl: bool
    timeout_seconds: int


def _parse_bool(v: str | None, default: bool) -> bool:
    if v is None or str(v).strip() == "":
        return default
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


def load_settings() -> Settings:
    load_dotenv()

    base_url = (os.getenv("SDP_API_BASE_URL") or "").strip()
    token = (os.getenv("SDP_AUTH_TOKEN") or "").strip()
    portal_id = (os.getenv("SDP_PORTAL_ID") or "").strip() or None
    verify_ssl = _parse_bool(os.getenv("SDP_VERIFY_SSL"), default=True)
    timeout_s_raw = (os.getenv("SDP_TIMEOUT_SECONDS") or "30").strip()

    if not base_url:
        raise SettingsError("Falta SDP_API_BASE_URL en variables de entorno.")
    if not token:
        raise SettingsError("Falta SDP_AUTH_TOKEN en variables de entorno.")

    try:
        timeout_s = int(timeout_s_raw)
    except ValueError as e:
        raise SettingsError("SDP_TIMEOUT_SECONDS debe ser un entero.") from e

    return Settings(
        api_base_url=base_url.rstrip("/"),
        auth_token=token,
        portal_id=portal_id,
        verify_ssl=verify_ssl,
        timeout_seconds=timeout_s,
    )
