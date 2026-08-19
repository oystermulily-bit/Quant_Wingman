"""
train_file.py — 从单个 Parquet K 线文件训练

用法:
    python train_file.py --data-file D:\\K线数据\\AAPL_H1.parquet

文件名格式: {品种}_{周期}.parquet，例如 AAPL_H1.parquet、US30.cash_H1.parquet
"""
from __future__ import annotations

import glob as _glob
import json
import pathlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.train_logging import configure_train_stdio

configure_train_stdio()

from config import Config
from data_pipeline.parquet_manager import ParquetDataManager, inspect_parquet_file
from model_core.config import ModelConfig
from model_core.engine import W1ngmanEngine
from model_core.holdout import (
    DevelopmentDataView,
)
from factor_research.release_audit import audit_engine_release
from model_core.vocab import VOCAB_VERSION


def train_from_file(data_file: str, *, from_scratch: bool = False) -> W1ngmanEngine | None:
    info = inspect_parquet_file(data_file)
    symbol = info["symbol"]
    timeframe = info["timeframe"]

    print(f"\n{'='*60}")
    print(f"  RD-Agent 文件训练 — {info['filename']}")
    print(f"{'='*60}")
    print(f"  品种: {symbol}")
    print(f"  周期: {timeframe}")
    print(f"  数据: 强制离线 Parquet（不连接 MT5）")
    print(f"  文件: {Path(data_file).resolve()}")
    print(f"  RD-Agent 研发轮数: {ModelConfig.TRAIN_STEPS}")
    print(f"  K线数: {info['bars']}")
    print(f"  模式: {'重新训练（从头）' if from_scratch else '自动续训'}")
    print(f"{'='*60}")

    try:
        mgr = ParquetDataManager(data_file)
        mgr.load()
        T = mgr.raw_dict["open"].shape[1]
        print(f"  数据加载成功，共 {T} 根K线")
    except Exception as e:
        print(f"  [错误] 数据加载失败: {e}")
        return None

    development_mgr = DevelopmentDataView(mgr, ModelConfig.HOLDOUT_FRACTION)
    holdout_spec = development_mgr.spec
    print(
        f"  数据隔离: 研发[0,{holdout_spec.split_index}) "
        f"共{holdout_spec.development_bars}根；最后10% Holdout"
        f"[{holdout_spec.split_index},{holdout_spec.total_bars}) "
        f"共{holdout_spec.holdout_bars}根"
    )

    engine = W1ngmanEngine(data_manager=development_mgr, target_symbol=symbol)
    engine.timeframe = timeframe
    engine.data_file = str(Path(data_file).resolve())
    engine.mode = "parquet_file"
    engine.train_steps = ModelConfig.TRAIN_STEPS
    engine.data_protocol = holdout_spec.protocol
    engine.holdout_fraction = holdout_spec.fraction
    engine.holdout_split_index = holdout_spec.split_index
    engine.full_bars = holdout_spec.total_bars

    ckpt_pattern = str(
        pathlib.Path("checkpoints") / f"ckpt_{symbol}_rd_step_*.pt"
    )
    ckpt_files = sorted(_glob.glob(ckpt_pattern))
    start_step = 0

    if from_scratch:
        removed = 0
        for p in ckpt_files:
            try:
                pathlib.Path(p).unlink(missing_ok=True)
                removed += 1
            except OSError as e:
                print(f"  [警告] 无法删除检查点 {p}: {e}")
        hist_path = pathlib.Path(f"training_history_{symbol}.json")
        if hist_path.exists():
            try:
                hist_path.unlink()
            except OSError:
                pass
        print(f"  [重新训练] 已清除 {removed} 个检查点，从第 0 步开始")
        ckpt_files = []
    elif ckpt_files:
        # Descend until a checkpoint with the same holdout protocol/split is found.
        for latest in reversed(ckpt_files):
            try:
                start_step = engine.load_checkpoint(latest)
                print(f"  [续训] 从 {latest} 恢复，起始步={start_step}")
                break
            except Exception as e:
                print(f"  [跳过] 不兼容检查点 {latest}: {e}")

    if start_step >= ModelConfig.TRAIN_STEPS:
        print(f"  [完成] {symbol} 已完成全部 {ModelConfig.TRAIN_STEPS} 步，跳过训练")
        frozen_formula = _save_strategy(engine, symbol, timeframe, data_file)
        _run_frozen_holdout(engine, mgr, holdout_spec, symbol, frozen_formula)
        return engine

    if start_step == 0 and not from_scratch:
        hist_path = pathlib.Path(f"training_history_{symbol}.json")
        if hist_path.exists():
            hist_path.unlink()
        print("  [新训] 从第 0 步开始")

    if start_step > 0:
        engine._save_training_history_live()

    engine.train(start_step=start_step)
    frozen_formula = _save_strategy(engine, symbol, timeframe, data_file)
    _run_frozen_holdout(engine, mgr, holdout_spec, symbol, frozen_formula)
    return engine


def _run_frozen_holdout(engine, full_manager, spec, symbol: str, formula) -> None:
    """Run or reuse the immutable release audit; never feed it into selection."""
    if formula is None:
        print("  [Holdout] 无冻结公式，未执行独立测试")
        return
    outcome = audit_engine_release(engine, full_manager, spec, symbol)
    result = outcome.result
    engine.holdout_report = outcome.report_path
    _attach_holdout_metadata(symbol, formula, engine.holdout_report)
    if result is None:
        print(
            f"  [Holdout] 发布审计状态={outcome.status}；"
            "同一发布快照不会重复读取 Holdout"
        )
        return
    print(
        f"  [Holdout] {'复用既有' if outcome.reused else '已执行'}冻结SOTA发布审计："
        f"Sharpe={result['sharpe']:.3f} "
        f"累计logPnL={result['total_log_pnl']:.4f}；未用于选优"
    )
    print(f"  [Holdout] release_id={outcome.release_id} 报告: {outcome.report_path}")


def _attach_holdout_metadata(symbol: str, formula: list[int], report: str) -> None:
    """Attach the report only when it describes the formula stored on disk."""
    path = pathlib.Path("strategies") / f"best_{symbol}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("formula") != list(formula):
            raise ValueError("strategy changed before holdout metadata update")
        data["holdout_report"] = report
        data["holdout_used_for_selection"] = False
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"  [警告] Holdout元数据未能写回策略文件: {exc}")


def _save_strategy(
    engine: W1ngmanEngine, symbol: str, timeframe: str, data_file: str
) -> list[int] | None:
    path = pathlib.Path("strategies") / f"best_{symbol}.json"
    path.parent.mkdir(exist_ok=True)
    if engine.best_formula is None:
        print("  [策略] 本次没有有效公式，未写入策略文件")
        return None
    # 若同一数据协议的磁盘策略分数更高，不要用更弱结果覆盖
    if path.exists() and engine.best_formula is not None:
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            old_score = old.get("best_score")
            same_backend = old.get("generator_backend") == ModelConfig.GENERATOR_BACKEND
            same_protocol = old.get("data_protocol") == getattr(engine, "data_protocol", None)
            research = getattr(engine, "factor_research", None)
            current_research_protocol = (
                research.validation_protocol
                if research is not None
                else getattr(engine, "research_protocol", None)
            )
            same_research_protocol = (
                old.get("research_protocol") == current_research_protocol
            )
            if (
                same_backend
                and same_protocol
                and same_research_protocol
                and old_score is not None
                and float(old_score) > float(engine.best_score)
            ):
                print(
                    f"  [策略] 保留磁盘更优结果 {float(old_score):.4f} "
                    f"> 本次 {float(engine.best_score):.4f}，未覆盖 {path}"
                )
                merged = dict(old)
                for key, val in (
                    ("timeframe", timeframe),
                    ("data_file", str(Path(data_file).resolve())),
                    ("mode", "parquet_file"),
                    ("train_steps", ModelConfig.TRAIN_STEPS),
                ):
                    if val is not None and not merged.get(key):
                        merged[key] = val
                if merged != old:
                    path.write_text(
                        json.dumps(merged, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    print(f"  [策略] 已补全数据路径等元数据: {path}")
                formula = old.get("formula")
                return list(formula) if isinstance(formula, list) else None
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass
    data = {
        "vocab_version": VOCAB_VERSION,
        "generator_backend": ModelConfig.GENERATOR_BACKEND,
        "symbol": symbol,
        "timeframe": timeframe,
        "data_file": str(Path(data_file).resolve()),
        "mode": "parquet_file",
        "formula": engine.best_formula,
        "formula_decoded": engine._decode_formula(engine.best_formula)
        if engine.best_formula
        else None,
        "best_score": engine.best_score,
        "train_steps": ModelConfig.TRAIN_STEPS,
        "data_protocol": getattr(engine, "data_protocol", None),
        "holdout_fraction": getattr(engine, "holdout_fraction", None),
        "holdout_split_index": getattr(engine, "holdout_split_index", None),
        "full_bars": getattr(engine, "full_bars", None),
        "holdout_report": getattr(engine, "holdout_report", None),
        "holdout_used_for_selection": False,
    }
    if engine.factor_research is not None:
        data["research_protocol"] = engine.factor_research.validation_protocol
        data["sota"] = engine.factor_research.status()
        data["best_factor_id"] = getattr(engine, "best_factor_id", None)
    elif getattr(engine, "research_protocol", None):
        data["research_protocol"] = engine.research_protocol
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  策略已保存: {path}")
    return list(engine.best_formula)


if __name__ == "__main__":
    ModelConfig.REWARD_MODE = "ftmo"

    if "--rounds" in sys.argv:
        rounds_idx = sys.argv.index("--rounds")
        if rounds_idx + 1 >= len(sys.argv):
            print("错误: --rounds 后需要填写 1 到 10000 之间的整数")
            sys.exit(2)
        try:
            requested_rounds = int(sys.argv[rounds_idx + 1])
        except ValueError:
            print("错误: --rounds 必须是整数")
            sys.exit(2)
        if not 1 <= requested_rounds <= 10000:
            print("错误: --rounds 必须在 1 到 10000 之间")
            sys.exit(2)
        ModelConfig.TRAIN_STEPS = requested_rounds

    # The web launcher passes this argument explicitly. Reject every other value
    # before loading data or creating W1ngmanEngine, so a stale configuration can
    # never make a web-started job fall through to the legacy REINFORCE branch.
    requested_backend = ModelConfig.GENERATOR_BACKEND
    if "--generator-backend" in sys.argv:
        backend_idx = sys.argv.index("--generator-backend")
        if backend_idx + 1 >= len(sys.argv):
            print("错误: --generator-backend 后需要生成器名称")
            sys.exit(2)
        requested_backend = sys.argv[backend_idx + 1].strip().lower()
    if requested_backend != "rd_agent":
        print(
            "错误: 此训练入口已锁定为 RD-Agent，拒绝启动旧 W1ngman/REINFORCE "
            f"生成器: {requested_backend!r}"
        )
        sys.exit(2)
    ModelConfig.GENERATOR_BACKEND = "rd_agent"
    print("[生成器确认] backend=rd_agent；旧 W1ngman/REINFORCE 已禁用")

    if "--data-file" not in sys.argv:
        print("用法: python train_file.py --data-file PATH\\TO\\SYMBOL_TF.parquet [--from-scratch]")
        print("示例: python train_file.py --data-file D:\\K线数据\\AAPL_H1.parquet")
        sys.exit(1)

    idx = sys.argv.index("--data-file")
    if idx + 1 >= len(sys.argv):
        print("错误: --data-file 后需要文件路径")
        sys.exit(1)

    data_file = sys.argv[idx + 1]
    from_scratch = "--from-scratch" in sys.argv
    t0 = time.time()
    eng = train_from_file(data_file, from_scratch=from_scratch)
    elapsed = time.time() - t0

    if eng:
        sym = eng.target_symbol or "?"
        print(f"\n<<< [{sym}] 训练完成: 最优分数={eng.best_score:.4f}，耗时 {elapsed/3600:.2f} 小时")
        if eng.best_formula:
            print(f"    {eng._decode_formula(eng.best_formula)}")
    else:
        print("\n<<< 训练失败")
        sys.exit(1)
