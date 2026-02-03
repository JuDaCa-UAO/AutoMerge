from __future__ import annotations

import streamlit as st

from servicedesk_merge.auth_store import encryption_available, get_token, save_token

_LANG = {
    "es": {
        "header": "Login / Sesion",
        "user": "Usuario",
        "token": "Token ServiceDesk (authtoken)",
        "remember": "Recordar en este equipo",
        "save": "Guardar token y entrar",
        "logout": "Cerrar sesion",
        "connected_as": "Conectado como",
        "token_loaded": "Token cargado",
        "yes": "Si",
        "no": "No",
        "user_required": "Debes ingresar un usuario.",
        "token_required": "Debes pegar tu token de ServiceDesk.",
        "loaded_from_device": "Token cargado desde este equipo.",
        "enc_missing": "No hay clave de cifrado; el token no se guardara en este equipo.",
    },
    "en": {
        "header": "Login / Session",
        "user": "User",
        "token": "ServiceDesk Token (authtoken)",
        "remember": "Remember on this device",
        "save": "Save token and sign in",
        "logout": "Sign out",
        "connected_as": "Connected as",
        "token_loaded": "Token loaded",
        "yes": "Yes",
        "no": "No",
        "user_required": "Please enter a username.",
        "token_required": "Please paste your ServiceDesk token.",
        "loaded_from_device": "Token loaded from this device.",
        "enc_missing": "Missing encryption key; token will not be saved on this device.",
    },
}


def _t(key: str) -> str:
    lang = st.session_state.get("ui_language", "es")
    return _LANG.get(lang, _LANG["es"]).get(key, key)


def _clear_auth_state() -> None:
    for k in (
        "auth_user",
        "auth_token",
        "auth_validated_for",
        "auth_auto_loaded_for",
    ):
        st.session_state.pop(k, None)
    st.session_state["auth_reset_inputs"] = True


def render_login_panel() -> tuple[str, str] | None:
    st.sidebar.markdown(f"### {_t('header')}")

    if st.session_state.get("auth_reset_inputs"):
        st.session_state["auth_username_input"] = ""
        st.session_state["auth_token_input"] = ""
        st.session_state["auth_reset_inputs"] = False

    current_user = st.session_state.get("auth_user", "")
    current_token = st.session_state.get("auth_token", "")

    user_label = current_user if current_user else "-"
    token_label = _t("yes") if current_token else _t("no")
    st.sidebar.write(f"{_t('connected_as')}: {user_label}")
    st.sidebar.write(f"{_t('token_loaded')}: {token_label}")

    username = st.sidebar.text_input(_t("user"), key="auth_username_input")
    token_input = st.sidebar.text_input(_t("token"), type="password", key="auth_token_input")
    remember = st.sidebar.checkbox(_t("remember"), value=True, key="auth_remember")

    col_a, col_b = st.sidebar.columns(2)
    save_click = col_a.button(_t("save"), type="primary")
    logout_click = col_b.button(_t("logout"))

    if logout_click:
        _clear_auth_state()
        st.sidebar.info(_t("token_loaded") + f": {_t('no')}")
        return None

    if save_click:
        user = str(username or "").strip()
        if not user:
            st.sidebar.error(_t("user_required"))
            return None

        token_value = str(token_input or "").strip()
        if not token_value:
            token_value = get_token(user) or ""
            if not token_value:
                st.sidebar.error(_t("token_required"))
                return None

        if remember:
            if not encryption_available():
                st.sidebar.warning(_t("enc_missing"))
            else:
                saved = save_token(user, token_value)
                if not saved:
                    st.sidebar.warning(_t("enc_missing"))

        st.session_state["auth_user"] = user
        st.session_state["auth_token"] = token_value
        st.session_state["auth_reset_inputs"] = True
        return user, token_value

    if username and not current_token and not current_user:
        auto_loaded_for = st.session_state.get("auth_auto_loaded_for")
        if auto_loaded_for != username:
            st.session_state["auth_auto_loaded_for"] = username
            stored = get_token(username)
            if stored:
                st.session_state["auth_user"] = str(username).strip()
                st.session_state["auth_token"] = stored
                st.sidebar.info(_t("loaded_from_device"))
                return st.session_state["auth_user"], st.session_state["auth_token"]

    if current_user and current_token:
        return current_user, current_token

    return None
