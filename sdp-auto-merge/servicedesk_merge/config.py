from __future__ import annotations

import os
from dataclasses import dataclass

try:
    import streamlit as st
except Exception:  # pragma: no cover - optional outside Streamlit
    st = None


class SettingsError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    api_base_url: str
    portal_id: str | None
    verify_ssl: bool
    timeout_seconds: int


def _parse_bool(v: str | None, default: bool) -> bool:
    if v is None or str(v).strip() == "":
        return default
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


def _get_setting(name: str) -> str | None:
    if st is not None:
        try:
            val = st.secrets.get(name)
            if val is not None:
                return val
        except Exception:
            pass
    return os.getenv(name)


def load_settings() -> Settings:
    base_url = (_get_setting("SDP_API_BASE_URL") or "").strip()
    portal_id = (_get_setting("SDP_PORTAL_ID") or "").strip() or None
    verify_ssl = _parse_bool(_get_setting("SDP_VERIFY_SSL"), default=True)
    timeout_s_raw = (_get_setting("SDP_TIMEOUT_SECONDS") or "30").strip()

    if not base_url:
        raise SettingsError("Falta SDP_API_BASE_URL en variables de entorno.")

    try:
        timeout_s = int(timeout_s_raw)
    except ValueError as e:
        raise SettingsError("SDP_TIMEOUT_SECONDS debe ser un entero.") from e

    return Settings(
        api_base_url=base_url.rstrip("/"),
        portal_id=portal_id,
        verify_ssl=verify_ssl,
        timeout_seconds=timeout_s,
    )
