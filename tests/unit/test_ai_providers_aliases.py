"""Alias routing for openclaw / openclaw_wb auto-detect."""
from __future__ import annotations

from web.ai_providers import (
    SILICONFLOW_BASE_URL,
    SILICONFLOW_MODEL,
    _alias_provider_from_key,
    resolve_provider,
)


def test_openclaw_wb_alias_wins_over_openclaw_prefix() -> None:
    assert _alias_provider_from_key("openclaw_wb") == "openclaw_wb"
    assert _alias_provider_from_key("openclaw_wb/auto") == "openclaw_wb"
    assert _alias_provider_from_key("OpenClaw_WB") == "openclaw_wb"


def test_openclaw_alias() -> None:
    assert _alias_provider_from_key("openclaw") == "openclaw"
    assert _alias_provider_from_key("openclaw/main") == "openclaw"
    assert _alias_provider_from_key("deepseek-key") is None
    assert _alias_provider_from_key("") is None


def test_resolve_siliconflow_provider() -> None:
    resolved = resolve_provider("siliconflow", "sk-test")
    assert resolved.provider == "siliconflow"
    assert resolved.api_key == "sk-test"
    assert resolved.base_url == SILICONFLOW_BASE_URL
    assert resolved.model == SILICONFLOW_MODEL


def test_siliconflow_requires_key() -> None:
    import pytest

    with pytest.raises(ValueError, match="硅基流动 API Key"):
        resolve_provider("siliconflow", "")
