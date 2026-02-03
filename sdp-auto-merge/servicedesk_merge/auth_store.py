from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

_STORE_DIR = Path(__file__).resolve().parent.parent / ".secrets"
_STORE_FILE = _STORE_DIR / "user_tokens.json"


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


def load_tokens() -> Dict[str, str]:
    if not _STORE_FILE.exists():
        return {}
    try:
        raw = _STORE_FILE.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    out: Dict[str, str] = {}
    for key, value in data.items():
        if value is None:
            continue
        out[str(key)] = str(value)
    return out


def save_token(username: str, token: str) -> None:
    user = str(username or "").strip()
    value = str(token or "").strip()
    if not user or not value:
        return

    _ensure_store_dir()
    data = load_tokens()
    data[user] = value
    _STORE_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
    _set_permissions(_STORE_FILE, is_dir=False)


def get_token(username: str) -> str | None:
    user = str(username or "").strip()
    if not user:
        return None
    return load_tokens().get(user)


def delete_token(username: str) -> None:
    user = str(username or "").strip()
    if not user:
        return
    data = load_tokens()
    if user in data:
        data.pop(user, None)
        if not data:
            try:
                _STORE_FILE.unlink()
            except Exception:
                pass
            return
        _ensure_store_dir()
        _STORE_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
        _set_permissions(_STORE_FILE, is_dir=False)
