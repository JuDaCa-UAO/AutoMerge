from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

try:
    from cryptography.fernet import Fernet, InvalidToken
except Exception:  # pragma: no cover - optional dependency
    Fernet = None
    InvalidToken = Exception

_STORE_DIR = Path(__file__).resolve().parent.parent / ".secrets"
_STORE_FILE = _STORE_DIR / "user_tokens.json"
_STORE_FORMAT = 1
_ENV_KEY_NAMES = ("SDP_TOKEN_ENC_KEY", "TOKEN_ENC_KEY")


def _set_permissions(path: Path, *, is_dir: bool) -> None:
    mode = 0o700 if is_dir else 0o600
    try:
        os.chmod(path, mode)
    except Exception:
        # Best-effort on Windows; ACLs may still allow broader access.
        pass


def _ensure_store_dir() -> None:
    _STORE_DIR.mkdir(parents=True, exist_ok=True)
    _set_permissions(_STORE_DIR, is_dir=True)


def _load_raw_tokens() -> Dict[str, str]:
    if not _STORE_FILE.exists():
        return {}
    try:
        raw = _STORE_FILE.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    if isinstance(data.get("tokens"), dict):
        tokens = data.get("tokens") or {}
        return {str(k): str(v) for k, v in tokens.items() if v is not None}
    return {str(k): str(v) for k, v in data.items() if v is not None}


def _get_key() -> bytes | None:
    if Fernet is None:
        return None
    key = ""
    for name in _ENV_KEY_NAMES:
        key = (os.getenv(name) or "").strip()
        if key:
            break
    if not key:
        return None
    try:
        key_bytes = key.encode("utf-8")
        Fernet(key_bytes)
        return key_bytes
    except Exception:
        return None


def encryption_available() -> bool:
    return _get_key() is not None


def _encrypt(token: str) -> str | None:
    key = _get_key()
    if key is None or Fernet is None:
        return None
    f = Fernet(key)
    return f.encrypt(token.encode("utf-8")).decode("utf-8")


def _decrypt(value: str) -> str | None:
    key = _get_key()
    if key is None or Fernet is None:
        return None
    f = Fernet(key)
    try:
        return f.decrypt(value.encode("utf-8")).decode("utf-8")
    except InvalidToken:
        return None


def load_tokens() -> Dict[str, str]:
    raw_tokens = _load_raw_tokens()
    if not raw_tokens:
        return {}
    out: Dict[str, str] = {}
    for key, value in raw_tokens.items():
        if not isinstance(value, str):
            continue
        token = _decrypt(value)
        if token:
            out[str(key)] = token
    return out


def save_token(username: str, token: str) -> bool:
    user = str(username or "").strip()
    value = str(token or "").strip()
    if not user or not value:
        return False

    encrypted = _encrypt(value)
    if not encrypted:
        return False

    _ensure_store_dir()
    data = _load_raw_tokens()
    data[user] = encrypted
    payload = {"_format": _STORE_FORMAT, "tokens": data}
    _STORE_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    _set_permissions(_STORE_FILE, is_dir=False)
    return True


def get_token(username: str) -> str | None:
    user = str(username or "").strip()
    if not user:
        return None
    return load_tokens().get(user)


def delete_token(username: str) -> None:
    user = str(username or "").strip()
    if not user:
        return
    data = _load_raw_tokens()
    if user in data:
        data.pop(user, None)
        if not data:
            try:
                _STORE_FILE.unlink()
            except Exception:
                pass
            return
        _ensure_store_dir()
        payload = {"_format": _STORE_FORMAT, "tokens": data}
        _STORE_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        _set_permissions(_STORE_FILE, is_dir=False)
