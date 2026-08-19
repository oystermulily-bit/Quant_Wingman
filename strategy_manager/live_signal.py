"""实时信号计算：因子公式 + 实时 K 线 → 方向 + 强度。

与回测走完全相同的计算链（compute_features → StackVM → tanh → 阈值），
保证实时信号与回测/训练目标一致。信号取最后一根已收盘 bar。
"""
from __future__ import annotations

import math
from typing import Any

import torch

from model_core.features import MT5FeatureEngineer
from model_core.vm import StackVM

# 默认无信号阈值（与 signal.MIN_TRADE_EXPOSURE 一致）；实际值在调用时从 Config 读取
_DEFAULT_MIN_EXPOSURE = 0.05
# 默认最小 bar 数；实际值在调用时从 Config 读取（保证训练-实盘一致）
# 训练侧 Config.MIN_BARS=3000，特征侧 _NORM_WINDOW=200，EMA_26 的 2*w_full=360
# 实盘必须 ≥ max(Config.MIN_BARS, 2 * max_ema_w_full) 才能避免 train-serve skew
_DEFAULT_MIN_BARS = 500


def _min_trade_exposure() -> float:
    """调用时从 Config 读取，保证与 signal.py / 回测一致（避免热更新不同步）。"""
    try:
        from config import Config
        return float(getattr(Config, "MIN_TRADE_EXPOSURE", _DEFAULT_MIN_EXPOSURE))
    except Exception:  # noqa: BLE001
        return _DEFAULT_MIN_EXPOSURE


def _min_bars() -> int:
    """调用时从 Config 读取最小 bar 数，与训练侧 Config.MIN_BARS 对齐。"""
    try:
        from config import Config
        # 实盘需要：特征 warm-up（_NORM_WINDOW=200）+ EMA_26 递推稳定（2*w_full=360）
        # 取 max(Config.MIN_BARS, 500) 保证数值路径与训练一致
        return max(int(getattr(Config, "MIN_BARS", _DEFAULT_MIN_BARS)), _DEFAULT_MIN_BARS)
    except Exception:  # noqa: BLE001
        return _DEFAULT_MIN_BARS


# 向后兼容：暴露模块级常量（实际逻辑用 _min_bars() 调用）
MIN_BARS = _DEFAULT_MIN_BARS

_VM = StackVM()

DIR_LONG = "LONG"
DIR_SHORT = "SHORT"
DIR_FLAT = "FLAT"


def min_exposure() -> float:
    return _min_trade_exposure()


def evaluate_signal(
    formula: list[int],
    raw_dict: dict[str, Any],
    require_closed: bool = True,
    server_time: int | None = None,
    bar_seconds: int = 3600,
) -> dict[str, Any]:
    """在实时 K 线上计算因子信号。

    Args:
        formula:       策略因子的 token 序列。
        raw_dict:      {open,high,low,close,volume,time} torch 张量 [1, T]，升序。
                       time 字段为 Unix 秒（int64）。
        require_closed: 是否校验最后一根 bar 已收盘（避免 tick 抖动导致信号翻转）。
        server_time:   当前服务器 Unix 秒。若提供且 require_closed=True，
                       会校验 raw_dict["time"][0,-1] < server_time - bar_seconds*0.5。
        bar_seconds:   一个 bar 的秒数（H1=3600），用于收盘校验。

    Returns:
        dict：state / direction / strength / factor_value / position / bars_used / message
    """
    close = raw_dict.get("close")
    if close is None or close.ndim != 2:
        return {"state": "error", "message": "行情数据格式无效"}

    n_bars = int(close.shape[1])
    min_bars_required = _min_bars()
    if n_bars < min_bars_required:
        return {
            "state": "insufficient",
            "bars_used": n_bars,
            "message": f"历史 bar 不足（{n_bars}/{min_bars_required}），无法稳定计算特征",
        }

    # 收盘校验：若启用 require_closed 且提供 server_time，检查最后一根 bar 是否已收盘
    # 避免传入未收盘的当前 bar 导致信号随 tick 抖动反复翻转
    if require_closed and server_time is not None:
        time_tensor = raw_dict.get("time")
        if time_tensor is not None:
            try:
                last_bar_time = int(time_tensor[0, -1].item())
                # 最后一根 bar 的开盘时间 + bar_seconds 即为收盘时间
                # 若当前时间距收盘不足半个 bar，认为尚未收盘，拒绝出信号
                if server_time - last_bar_time < bar_seconds - bar_seconds // 2:
                    return {
                        "state": "pending",
                        "bars_used": n_bars,
                        "message": (
                            f"最后一根 bar 未收盘（last_bar_time={last_bar_time}, "
                            f"server_time={server_time}），等待收盘后再计算信号"
                        ),
                    }
            except Exception:  # noqa: BLE001
                # 时间字段异常时降级为不校验
                pass

    try:
        feats = MT5FeatureEngineer.compute_features(raw_dict)  # [1, F, T]
    except Exception as exc:  # noqa: BLE001
        return {"state": "error", "bars_used": n_bars, "message": f"特征计算失败: {exc}"}

    try:
        factor = _VM.execute([int(t) for t in formula], feats)  # [1, T] or None
    except Exception as exc:  # noqa: BLE001
        return {"state": "error", "bars_used": n_bars, "message": f"公式执行失败: {exc}"}

    if factor is None or factor.ndim != 2 or factor.shape[1] == 0:
        return {"state": "error", "bars_used": n_bars, "message": "公式无有效输出"}

    factor_last = float(factor[0, -1])
    if not math.isfinite(factor_last):
        return {"state": "error", "bars_used": n_bars, "message": "因子值非有限"}

    position = math.tanh(factor_last)          # 连续仓位 [-1, 1]
    strength = abs(position)                    # 信号强度 [0, 1]
    thr = _min_trade_exposure()

    if position >= thr:
        direction = DIR_LONG
    elif position <= -thr:
        direction = DIR_SHORT
    else:
        direction = DIR_FLAT

    return {
        "state": "ok",
        "direction": direction,
        "strength": round(strength, 4),
        "position": round(position, 4),
        "factor_value": round(factor_last, 6),
        "threshold": thr,
        "bars_used": n_bars,
        "message": "",
    }
