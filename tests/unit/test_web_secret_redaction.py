from __future__ import annotations

import re

import pytest
from fastapi import HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from web import app as web_app


SECRET_SETTINGS = {
    "ai_provider": "siliconflow",
    "ai_api_key": "sk-do-not-return",
    "feishu_enabled": True,
    "feishu_webhook_url": "https://example.invalid/private-hook",
    "feishu_secret": "private-signing-secret",
    "tqsdk_user": "private-user",
    "tqsdk_password": "private-password",
    "last_data_file": "",
    "debug_mode": False,
}


def test_cors_does_not_expose_local_credential_proxy_to_arbitrary_origins() -> None:
    middleware = next(
        item for item in web_app.app.user_middleware if item.cls is CORSMiddleware
    )
    assert "*" not in middleware.kwargs["allow_origins"]
    pattern = middleware.kwargs["allow_origin_regex"]
    assert re.fullmatch(pattern, "http://127.0.0.1:8765")
    assert re.fullmatch(pattern, "http://localhost:8876")
    assert not re.fullmatch(pattern, "https://attacker.example")


def test_settings_endpoint_never_returns_stored_credentials(monkeypatch) -> None:
    monkeypatch.setattr(web_app, "load_settings", lambda: dict(SECRET_SETTINGS))

    payload = web_app.api_get_settings()

    assert payload["ai_api_key"] == ""
    assert payload["feishu_webhook_url"] == ""
    assert payload["feishu_secret"] == ""
    assert payload["tqsdk_user"] == ""
    assert payload["tqsdk_password"] == ""
    assert payload["has_api_key"] is True
    assert payload["has_feishu_webhook"] is True
    assert payload["has_feishu_secret"] is True
    assert payload["has_tqsdk_credentials"] is True


def test_training_import_is_loopback_only() -> None:
    local = Request({"type": "http", "client": ("127.0.0.1", 1234)})
    remote = Request({"type": "http", "client": ("192.168.1.20", 1234)})

    web_app._require_loopback_client(local)
    with pytest.raises(HTTPException) as exc:
        web_app._require_loopback_client(remote)
    assert exc.value.status_code == 403


def test_config_marks_legacy_scope_without_returning_key(monkeypatch) -> None:
    monkeypatch.setattr(web_app, "load_settings", lambda: dict(SECRET_SETTINGS))
    monkeypatch.setattr(
        web_app,
        "_strategy_context",
        lambda: {"last_strategy_file": None, "strategy_file": None},
    )
    monkeypatch.setattr(
        web_app,
        "debug_snapshot",
        lambda _lines: {"server_log": "server.log", "error_log": "error.log"},
    )

    payload = web_app.api_config()

    assert payload["ai_api_key"] == ""
    assert payload["has_api_key"] is True
    assert payload["training_scope"] == "LEGACY_SINGLE_SYMBOL_PARQUET"
    assert payload["csi300_panel_engine_connected"] is False
    assert payload["csi300_conclusion_status"] == "NOT_AVAILABLE_ADAPTER_NOT_CONNECTED"


def test_feishu_endpoints_return_presence_only(monkeypatch) -> None:
    monkeypatch.setattr(web_app, "load_settings", lambda: dict(SECRET_SETTINGS))
    monkeypatch.setattr(web_app, "save_settings", lambda _payload: dict(SECRET_SETTINGS))

    fetched = web_app.api_realtime_feishu_get()
    saved = web_app.api_realtime_feishu_put(web_app.FeishuSettingsRequest(enabled=True))

    for payload in (fetched, saved):
        assert payload["webhook_url"] == ""
        assert payload["secret"] == ""
        assert payload["has_webhook"] is True
        assert payload["has_secret"] is True
