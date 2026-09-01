"""Persisted UI settings for the training web console."""
from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = PROJECT_ROOT / "web_settings.json"
STRATEGIES_DIR = PROJECT_ROOT / "strategies"

_DEFAULT = {
    "last_data_file": "",
    "last_strategy_file": "",
    "debug_mode": False,
    "ai_provider": "deepseek",
    "ai_api_key": "",
    # 回测单边成本（单位 %）：手续费 0.02% + 滑点 0.01% ≈ 常见加密货币轻度成本
    "bt_commission_pct": 0.02,
    "bt_slippage_pct": 0.01,
    # 实时分析监控清单：[{source, symbol, timeframe, strategy_file}, ...]
    "realtime_watches": [],
    # 飞书机器人（信号转折提醒，仅文本）
    "feishu_enabled": False,
    "feishu_webhook_url": "",
    "feishu_secret": "",
}

# Secrets may be supplied by the UI for the lifetime of the current process, but
# are never written to web_settings.json. Persistent configuration belongs in
# the process environment or the operating system's secret manager.
_SECRET_ENV = {
    "ai_api_key": "W1NGMAN_AI_API_KEY",
    "feishu_webhook_url": "W1NGMAN_FEISHU_WEBHOOK_URL",
    "feishu_secret": "W1NGMAN_FEISHU_SECRET",
}
_SECRET_KEYS = frozenset(_SECRET_ENV)
_PERSISTED_KEYS = frozenset(_DEFAULT) - _SECRET_KEYS
# Dropped tqsdk / domestic-futures credentials; still strip them from old JSON.
_RETIRED_SECRET_KEYS = frozenset({"tqsdk_user", "tqsdk_password"})
_RETIRED_WATCH_SOURCES = frozenset({"domestic_futures"})


def _environment_secret(key: str) -> str:
    value = str(os.environ.get(_SECRET_ENV[key], "") or "").strip()
    if value or key != "ai_api_key":
        return value
    # Keep the existing provider-specific deployment variables compatible.
    return str(
        os.environ.get("SILICONFLOW_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or ""
    ).strip()


def _persisted_payload(settings: dict) -> dict:
    """Return the public/non-secret subset allowed on disk."""
    return {key: settings.get(key, _DEFAULT[key]) for key in _PERSISTED_KEYS}


def _write_persisted_settings(settings: dict) -> None:
    SETTINGS_PATH.write_text(
        json.dumps(_persisted_payload(settings), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _as_pct(value, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v < 0:
        return default
    return v


def _is_ephemeral_data_path(path: str) -> bool:
    """Stale pytest temp parquet paths (not valid training data)."""
    norm = str(path or "").replace("\\", "/").lower()
    if "pytest-of-" not in norm:
        return False
    return (
        "/appdata/local/temp/" in norm
        or norm.startswith("/tmp/")
        or "/temp/" in norm
    )


def _is_production_settings_path() -> bool:
    try:
        return SETTINGS_PATH.resolve() == (PROJECT_ROOT / "web_settings.json").resolve()
    except OSError:
        return False


def _is_usable_data_file(path: str) -> bool:
    p = Path(str(path or "").strip())
    return p.is_file() and p.suffix.lower() == ".parquet"


def _should_replace_last_data_file(path: str) -> bool:
    cur = str(path or "").strip()
    if not cur:
        return True
    if not Path(cur).is_file():
        return True
    return _is_ephemeral_data_path(cur)


def _data_file_from_strategy_json(path: str) -> str | None:
    p = Path(str(path or "").strip())
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    candidate = str(data.get("data_file") or "").strip()
    if _is_usable_data_file(candidate):
        return str(Path(candidate).resolve())
    return None


def _recover_last_data_file(current: dict) -> str:
    cur = str(current.get("last_data_file") or "").strip()
    if cur and not _should_replace_last_data_file(cur):
        return str(Path(cur).resolve())

    for strategy_path in (
        str(current.get("last_strategy_file") or "").strip(),
        *(str(p) for p in sorted(STRATEGIES_DIR.glob("best_*.json")) if p.is_file()),
    ):
        if not strategy_path:
            continue
        candidate = _data_file_from_strategy_json(strategy_path)
        if candidate:
            return candidate
    return cur


def load_settings() -> dict:
    data: dict = {}
    if SETTINGS_PATH.exists():
        try:
            loaded = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (json.JSONDecodeError, OSError):
            data = {}
    out = dict(_DEFAULT)
    out.update({k: v for k, v in data.items() if k in _PERSISTED_KEYS})
    for key in _SECRET_KEYS:
        out[key] = _environment_secret(key)
    out["debug_mode"] = bool(out.get("debug_mode", False))
    out["last_strategy_file"] = str(out.get("last_strategy_file") or "").strip()
    out["ai_provider"] = str(out.get("ai_provider") or "deepseek").strip().lower()
    if out["ai_provider"] not in ("deepseek", "siliconflow", "openclaw", "openclaw_wb"):
        out["ai_provider"] = "deepseek"
    out["ai_api_key"] = str(out.get("ai_api_key") or "").strip()
    out["bt_commission_pct"] = _as_pct(
        out.get("bt_commission_pct"), _DEFAULT["bt_commission_pct"]
    )
    out["bt_slippage_pct"] = _as_pct(
        out.get("bt_slippage_pct"), _DEFAULT["bt_slippage_pct"]
    )
    watches = out.get("realtime_watches")
    if not isinstance(watches, list):
        watches = []
    cleaned = []
    had_retired_watches = False
    for w in watches:
        if not isinstance(w, dict):
            continue
        src = str(w.get("source") or "").strip()
        sym = str(w.get("symbol") or "").strip()
        tf = str(w.get("timeframe") or "").strip()
        sf = str(w.get("strategy_file") or "").strip()
        if src in _RETIRED_WATCH_SOURCES:
            had_retired_watches = True
            continue
        if src and sym and tf and sf:
            cleaned.append(
                {"source": src, "symbol": sym, "timeframe": tf, "strategy_file": sf}
            )
    out["realtime_watches"] = cleaned
    out["feishu_enabled"] = bool(out.get("feishu_enabled", False))
    out["feishu_webhook_url"] = str(out.get("feishu_webhook_url") or "").strip()
    out["feishu_secret"] = str(out.get("feishu_secret") or "").strip()
    recovered = _recover_last_data_file(out)
    if recovered != out.get("last_data_file") and _is_production_settings_path():
        out["last_data_file"] = recovered
        if recovered:
            _write_persisted_settings(out)
    elif recovered != out.get("last_data_file"):
        out["last_data_file"] = recovered
    # One-way migration: strip credentials left by older versions immediately.
    if SETTINGS_PATH.exists() and (
        had_retired_watches
        or any(key in data for key in (_SECRET_KEYS | _RETIRED_SECRET_KEYS))
    ):
        _write_persisted_settings(out)
    return out


def save_settings(data: dict) -> dict:
    current = load_settings()
    for key, env_name in _SECRET_ENV.items():
        if key not in data:
            continue
        value = str(data[key] or "").strip()
        if value:
            os.environ[env_name] = value
        else:
            os.environ.pop(env_name, None)
        current[key] = value
    if "last_data_file" in data:
        path = str(data["last_data_file"] or "").strip()
        if (
            path
            and _is_ephemeral_data_path(path)
            and _is_production_settings_path()
        ):
            data = {k: v for k, v in data.items() if k != "last_data_file"}
        else:
            current["last_data_file"] = path
    if "last_strategy_file" in data:
        current["last_strategy_file"] = str(data["last_strategy_file"] or "").strip()
    if "debug_mode" in data:
        current["debug_mode"] = bool(data["debug_mode"])
    if "ai_provider" in data:
        provider = str(data["ai_provider"] or "deepseek").strip().lower()
        current["ai_provider"] = (
            provider
            if provider in ("deepseek", "siliconflow", "openclaw", "openclaw_wb")
            else "deepseek"
        )
    if "bt_commission_pct" in data:
        current["bt_commission_pct"] = _as_pct(
            data["bt_commission_pct"], _DEFAULT["bt_commission_pct"]
        )
    if "bt_slippage_pct" in data:
        current["bt_slippage_pct"] = _as_pct(
            data["bt_slippage_pct"], _DEFAULT["bt_slippage_pct"]
        )
    if "realtime_watches" in data:
        watches = data["realtime_watches"]
        if not isinstance(watches, list):
            watches = []
        cleaned = []
        for w in watches:
            if not isinstance(w, dict):
                continue
            src = str(w.get("source") or "").strip()
            sym = str(w.get("symbol") or "").strip()
            tf = str(w.get("timeframe") or "").strip()
            sf = str(w.get("strategy_file") or "").strip()
            if src in _RETIRED_WATCH_SOURCES:
                continue
            if src and sym and tf and sf:
                cleaned.append(
                    {
                        "source": src,
                        "symbol": sym,
                        "timeframe": tf,
                        "strategy_file": sf,
                    }
                )
        current["realtime_watches"] = cleaned
    if "feishu_enabled" in data:
        current["feishu_enabled"] = bool(data["feishu_enabled"])
    _write_persisted_settings(current)
    return current
