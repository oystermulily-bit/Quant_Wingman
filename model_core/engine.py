import heapq
import json
import os
import pathlib
import time

import torch

# ── 并行评估线程配置（必须在第一个张量操作前设置）──────────────────────────
# torch.set_num_interop_threads 一旦有任何张量 op 就锁定，所以紧跟 import torch。
# 策略：保持 intra_threads=full cores，让 PyTorch 内部
# 多线程处理大张量；同时开 8 workers 并行评估不同公式。
# PyTorch 的 intra-op 线程在执行期间会释放 GIL，允许多个 worker 真正并行。
_PHYS_CORES = os.cpu_count() or 4
_EVAL_WORKERS = min(_PHYS_CORES, 8)
_INTRA = _PHYS_CORES  # 保持满线程，不做 phys // workers
try:
    torch.set_num_interop_threads(_EVAL_WORKERS)
except RuntimeError:
    pass  # 已被锁定
torch.set_num_threads(_INTRA)

from tqdm import tqdm

from .config import ModelConfig
from .vm import StackVM
from .backtest import MT5Backtest, estimate_periods_per_year
from .vocab import FORMULA_VOCAB, VOCAB_VERSION, VocabVersionMismatchError  # task 12.2
from .rd_agent_generator import RDFormulaError, RDFormulaGenerator, RDFormulaLimits

# P3：冠军在场时间稳健性校验所需
try:
    from strategy_manager.signal import compute_target_positions_stateless
except ImportError:
    # 兼容无 strategy_manager 的测试环境
    def compute_target_positions_stateless(factors):  # type: ignore
        import torch as _torch
        return _torch.sign(_torch.tanh(factors))

try:
    from config import Config as _RootConfig
    _STRATEGY_FILE  = _RootConfig.STRATEGY_FILE
    _CHECKPOINT_DIR = pathlib.Path(getattr(_RootConfig, 'CHECKPOINT_DIR', 'checkpoints'))
except ImportError:
    _STRATEGY_FILE  = "best_mt5_strategy.json"
    _CHECKPOINT_DIR = pathlib.Path("checkpoints")


def _strategy_file_for_symbol(symbol: str | None) -> str:
    """返回该品种对应的策略文件路径。

    单品种训练时使用 strategies/best_{symbol}.json，
    多品种/未指定品种时回退到默认路径。
    """
    if symbol:
        return str(pathlib.Path("strategies") / f"best_{symbol}.json")
    return _STRATEGY_FILE


def _fallback_data_file_for_symbol(symbol: str) -> tuple[str | None, str | None]:
    """Read web_settings.json last_data_file when strategy JSON lacks data_file."""
    settings_path = pathlib.Path("web_settings.json")
    if not settings_path.exists():
        return None, None
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        last = str(settings.get("last_data_file") or "").strip()
    except (json.JSONDecodeError, OSError):
        return None, None
    if not last:
        return None, None
    p = pathlib.Path(last)
    if not p.exists():
        return None, None
    try:
        from data_pipeline.parquet_manager import inspect_parquet_file

        info = inspect_parquet_file(str(p.resolve()))
    except Exception:
        return str(p.resolve()), None
    if info.get("symbol") != symbol:
        return None, None
    return str(p.resolve()), info.get("timeframe")


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward 折叠构建
# ─────────────────────────────────────────────────────────────────────────────

def _build_walk_forward_folds(
    T: int,
    n_folds: int = 5,
    gap: int = 20,
    label_horizon: int = 2,
) -> list[dict]:
    """Build expanding walk-forward folds after reserving every purge gap.

    ``n_folds`` consists of one initial training block followed by
    ``n_folds - 1`` validation blocks.  The requested gap is never silently
    reduced, and trailing rows without observable labels are excluded.
    """
    if T <= 0 or n_folds < 2 or gap < 0 or label_horizon < 0:
        raise ValueError("invalid walk-forward arguments")

    usable_end = T - label_horizon
    validation_folds = n_folds - 1
    validation_size = (usable_end - validation_folds * gap) // n_folds
    if validation_size < 2:
        raise ValueError(
            "insufficient data for the requested walk-forward folds, "
            "purge gap, and label horizon"
        )

    initial_train_size = usable_end - validation_folds * (
        validation_size + gap
    )
    if initial_train_size < 2:
        raise ValueError("insufficient data for the initial training window")

    folds: list[dict] = []
    for index in range(validation_folds):
        train_end = initial_train_size + index * (validation_size + gap)
        val_start = train_end + gap
        val_end = val_start + validation_size
        folds.append(
            {
                "train_start": 0,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "gap": gap,
            }
        )
    return folds


def _repetition_penalty(formula: list[int]) -> float:
    if not formula:
        return 0.0
    penalty, count = 0.0, 1
    for i in range(1, len(formula)):
        if formula[i] == formula[i - 1]:
            count += 1
            if count >= 2:
                penalty += 0.3
        else:
            count = 1
    return penalty


def _current_research_protocol(n_folds: int) -> str:
    from factor_research.config import ResearchConfig
    from factor_research.factor_artifact import research_protocol_version

    return research_protocol_version(
        ResearchConfig(),
        pathlib.Path(__file__).resolve().parents[1],
        n_folds=n_folds,
        label_horizon=getattr(ModelConfig, "LABEL_HORIZON", 2),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ConstrainedSampler — 保证 100% 合法公式
# ─────────────────────────────────────────────────────────────────────────────
class W1ngmanEngine:
    def __init__(self, data_manager=None, use_lord_regularization=True,
                 lord_decay_rate=1e-3, lord_num_iterations=5, n_folds: int = 5,
                 target_symbol: str | None = None):
        self.data_manager  = data_manager
        self.n_folds       = n_folds
        self.target_symbol = target_symbol   # None = 多品种模式，str = 单品种模式
        self.generator_backend = ModelConfig.GENERATOR_BACKEND
        if self.generator_backend != "rd_agent":
            raise RuntimeError(
                "W1ngmanEngine 已锁定为 RD-Agent 公式生成器；"
                f"拒绝启动旧生成后端 {self.generator_backend!r}"
            )

        # Formula generation is RD-Agent-only. These compatibility attributes stay
        # empty so existing scoring/checkpoint consumers cannot instantiate or
        # restore the removed policy-gradient generator by accident.
        self.model = None
        self.opt = None
        self.use_lord = False
        self.lord_opt = None
        self.rank_monitor = None

        self.vm = StackVM()
        self.bt = MT5Backtest()

        self.best_score   = -float('inf')
        self.best_formula = None
        self._best_snapshot: dict | None = None

        self.training_history = {
            'step': [], 'avg_reward': [], 'best_score': [], 'val_score': [], 'stable_rank': []
        }
        self._restart_count      = 0
        self.factor_pool: list[tuple[float, int, torch.Tensor]] = []
        self._factor_pool_counter = 0

        # Elite Replay pool: (val_score, counter, formula_tokens, birth_step)
        self._elite_pool: list[tuple[float, int, list[int], int]] = []
        self._elite_counter = 0

        # 自适应噪声：记录 best 刷新步数
        self._best_update_step = 0
        self._stagnation_steps = 0

        # Fix 3: EMA reward baseline
        self._reward_ema: float | None = None
        self._reward_ema_step: int = 0

        # ── 并行评估线程池（CPU 利用率优化）──────────────────────────────
        self._eval_pool = None
        self._eval_workers = 1
        if ModelConfig.PARALLEL_EVAL:
            self._init_parallel_eval()

        self.rd_generator = RDFormulaGenerator(
            project_root=pathlib.Path(__file__).resolve().parents[1],
            model=ModelConfig.RD_AGENT_MODEL,
            limits=RDFormulaLimits(
                max_tokens=ModelConfig.RD_AGENT_MAX_TOKENS,
                max_depth=ModelConfig.RD_AGENT_MAX_DEPTH,
                max_operators=ModelConfig.RD_AGENT_MAX_OPERATORS,
                api_timeout_seconds=ModelConfig.RD_AGENT_API_TIMEOUT_SECONDS,
                round_timeout_seconds=ModelConfig.RD_AGENT_ROUND_TIMEOUT_SECONDS,
                api_retries=ModelConfig.RD_AGENT_API_RETRIES,
            ),
        )
        self.factor_research = None

    def _ensure_holdout_isolation(self) -> None:
        """Force every training entry point onto the prefix-only data view."""
        if self.data_manager is None:
            return
        expected = ModelConfig.HOLDOUT_PROTOCOL
        if getattr(self.data_manager, "data_protocol", None) == expected:
            return

        from .holdout import DevelopmentDataView

        full_manager = self.data_manager
        development = DevelopmentDataView(
            full_manager, fraction=ModelConfig.HOLDOUT_FRACTION
        )
        self._full_data_manager = full_manager
        self.data_manager = development
        self.data_protocol = development.spec.protocol
        self.holdout_fraction = development.spec.fraction
        self.holdout_split_index = development.spec.split_index
        self.full_bars = development.spec.total_bars
        tqdm.write(
            "[数据隔离] 已强制保留最后10% Holdout；"
            f"研发={development.spec.development_bars} bars，"
            f"Holdout={development.spec.holdout_bars} bars"
        )

    # ── 并行评估初始化 ──────────────────────────────────────────────────────

    def _init_parallel_eval(self):
        """初始化 ThreadPoolExecutor 用于并行公式评估。

        PyTorch CPU 算子会释放 GIL，多个 worker 线程可以真正并行执行
        vm.execute / bt.evaluate_fold 等纯张量计算。

        线程配置：
        - intra_threads = physical_cores（保持满，让大张量操作快）
        - inter_threads = workers（允许多 worker 并行调度）
        PyTorch 的 intra-op 线程池会自适应负载，不会真的 8×8=64 全跑满。
        """
        from concurrent.futures import ThreadPoolExecutor

        phys = _PHYS_CORES
        workers = ModelConfig.EVAL_WORKERS
        if workers <= 0:
            workers = min(phys, 8)
        intra = ModelConfig.EVAL_INTRA_THREADS
        if intra <= 0:
            intra = phys  # 保持满线程
        # 线程已在模块 import 时设置，这里只确认
        try:
            torch.set_num_threads(intra)
        except RuntimeError:
            pass

        self._eval_workers = workers
        self._eval_pool = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="w1ngman-eval",
        )
        print(f"[ParallelEval] workers={workers} intra_threads={intra} "
              f"(physical_cores={phys})  pool={'ON' if workers > 1 else 'OFF'}",
              flush=True)

    # ── 单条公式评估任务（线程安全）───────────────────────────────────────

    def _eval_formula_task(
        self,
        idx: int,
        fml: list[int],
        feat: torch.Tensor,
        t_ret: torch.Tensor,
        folds: list[dict],
        use_wf: bool,
        factor_pool_snapshot: list,
    ) -> dict:
        """评估单条公式（线程安全，可被多个 worker 并发调用）。

        所有共享对象（self.vm, self.bt）在评估路径上都是只读的；
        factor_pool 通过 snapshot 传入只读快照。
        """
        try:
            with torch.no_grad():
                execution = self.vm.execute_with_diagnostics(fml, feat)

            if execution is None:
                return {'idx': idx, 'status': 'none', 'reward': -5.0,
                        'val_score': -5.0, 'fml': fml}
            res = execution.values
            invalid_mask = execution.invalid_mask
            if res.std() < 1e-4:
                return {'idx': idx, 'status': 'const', 'reward': -2.0,
                        'val_score': -2.0, 'fml': fml}

            with torch.no_grad():
                if use_wf:
                    fold_tr, fold_vl, fold_ic = [], [], []
                    for fold in folds:
                        tr_sc, vl_sc = self.bt.evaluate_fold(
                            res, t_ret,
                            fold["train_start"], fold["train_end"],
                            fold["val_start"],   fold["val_end"],
                        )
                        ic_m, _ = W1ngmanEngine._compute_ic(
                            res[:, fold["train_start"]:fold["train_end"]],
                            t_ret[:, fold["train_start"]:fold["train_end"]],
                        )
                        tr_adj = W1ngmanEngine._apply_ic_gate(tr_sc, ic_m)
                        fold_tr.append(ModelConfig.W1NGMAN_REWARD_WEIGHT * tr_adj)
                        ic_v, _ = W1ngmanEngine._compute_ic(
                            res[:, fold["val_start"]:fold["val_end"]],
                            t_ret[:, fold["val_start"]:fold["val_end"]],
                        )
                        vl_adj = W1ngmanEngine._apply_ic_gate(vl_sc, ic_v)
                        fold_vl.append(vl_adj)
                        fold_ic.append(ic_m.item())
                    train_score = torch.stack(fold_tr).mean()
                    val_score = torch.stack(fold_vl).mean()
                    ic_i = sum(fold_ic) / len(fold_ic)
                else:
                    T_total = res.shape[1]
                    split_pt = max(int(T_total * 0.8), T_total - 100)
                    train_score, _ = self.bt.evaluate(res, {}, t_ret)
                    ic_m0, _ = W1ngmanEngine._compute_ic(res, t_ret)
                    train_score = W1ngmanEngine._apply_ic_gate(
                        ModelConfig.W1NGMAN_REWARD_WEIGHT * train_score, ic_m0
                    )
                    if split_pt < T_total - 1:
                        vl_sc, _ = self.bt.evaluate_fold(
                            res, t_ret, 0, split_pt, split_pt, T_total,
                        )
                        ic_v0, _ = W1ngmanEngine._compute_ic(
                            res[:, split_pt:], t_ret[:, split_pt:],
                        )
                        val_score = W1ngmanEngine._apply_ic_gate(vl_sc, ic_v0)
                    else:
                        val_score = train_score
                    ic_i = ic_m0.item()
                ic_full, ic_stab_full = W1ngmanEngine._compute_ic(res, t_ret)

            # 惩罚（纯函数，线程安全）
            reward = train_score
            val_score_out = val_score

            # 重复惩罚
            rp = _repetition_penalty(fml)
            if rp > 0:
                reward = reward - rp
                val_score_out = val_score_out - rp

            # 相关性惩罚（用 step 起始快照）
            if use_wf:
                _corr_slice = (folds[0]["train_start"], folds[0]["train_end"])
            else:
                _corr_slice = (0, max(int(res.shape[1] * 0.8), res.shape[1] - 100))
            reward = self._apply_corr_penalty(reward, res, _corr_slice)
            val_score_out = self._apply_corr_penalty(val_score_out, res, _corr_slice)

            return {
                'idx': idx, 'status': 'ok',
                'reward': reward.item() if isinstance(reward, torch.Tensor) else float(reward),
                'val_score': val_score_out.item() if isinstance(val_score_out, torch.Tensor) else float(val_score_out),
                'ic_full': ic_full.item(), 'ic_stab': ic_stab_full.item(),
                'ic_i': ic_i, 'res': res, 'invalid_mask': invalid_mask,
                'invalid_fraction': float(invalid_mask.float().mean().item()),
                'fml': fml,
            }
        except Exception as e:
            return {'idx': idx, 'status': 'error', 'reward': -5.0,
                    'val_score': -5.0, 'fml': fml,
                    'error': f'{type(e).__name__}: {e}'}

    # ── IC computation ────────────────────────────────────────────────────────

    @staticmethod
    def _compute_ic(factor: torch.Tensor, target_ret: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """时序 IC（每品种内部 factor[t] vs target_ret[t]）的均值与稳定性。

        ``DataManager.target_ret[t]`` 已表示信号日 ``t`` 收盘后、下一开盘
        执行的前视收益，因此不能再次移位。只在同索引的有限样本上计算，
        也为后续面板 Adapter 保留 NaN/停牌语义。
        """
        N, T = factor.shape
        if T < 2:
            z = torch.zeros(1, device=factor.device)
            return z, z

        ic_list = []
        for n in range(N):
            x = factor[n]
            y = target_ret[n]
            valid = torch.isfinite(x) & torch.isfinite(y)
            if valid.sum() < 2:
                continue
            x = x[valid]
            y = y[valid]
            xm = x - x.mean()
            ym = y - y.mean()
            sx = (xm ** 2).mean().sqrt()
            sy = (ym ** 2).mean().sqrt()
            if sx < 1e-6 or sy < 1e-6:
                continue
            ic = (xm * ym).mean() / (sx * sy + 1e-8)
            ic_list.append(ic)

        if not ic_list:
            z = torch.zeros(1, device=factor.device)
            return z, z

        ic_tensor = torch.stack(ic_list)
        ic_mean   = ic_tensor.mean()
        ic_stab   = (ic_mean / (ic_tensor.std(unbiased=False) + 1e-6)
                     if ic_tensor.numel() >= 2
                     else torch.zeros(1, device=factor.device))
        return ic_mean, ic_stab

    # ── IC gate: direction-based, dimension-agnostic ──────────────────────────

    @staticmethod
    def _apply_ic_gate(reward: torch.Tensor, ic_mean) -> torch.Tensor:
        """IC 门控：用 IC 符号而非量值调整 reward，完全规避量纲问题。
        IC > thresh  → reward × IC_GATE_MULT  (正向预测，奖励)
        IC < -thresh → reward × IC_NEG_MULT   (反向预测，惩罚)
        |IC| ≤ thresh→ 不修改                  (噪声区)
        """
        ic_val = ic_mean.item() if isinstance(ic_mean, torch.Tensor) else float(ic_mean)
        t = ModelConfig.IC_GATE_THRESH
        if ic_val > t:
            return reward * ModelConfig.IC_GATE_MULT
        elif ic_val < -t:
            return reward * ModelConfig.IC_NEG_MULT
        return reward


    # ── Elite pool ────────────────────────────────────────────────────────────

    @staticmethod
    def _dedup_elite_pool(
        pool: list[tuple[float, int, list[int], int]]
    ) -> list[tuple[float, int, list[int], int]]:
        """对精英池去重：相同 tokens 只保留得分最高的一条，重建最小堆。"""
        best: dict[str, tuple[float, int, list[int], int]] = {}
        for sc, cnt, toks, birth in pool:
            key = str(toks)
            if key not in best or sc > best[key][0]:
                best[key] = (sc, cnt, toks, birth)
        deduped = list(best.values())
        heapq.heapify(deduped)
        return deduped

    def _update_elite_pool(self, val_score: float, formula: list[int], step: int = 0) -> None:
        """维护精英公式池（最小堆，Top-ELITE_POOL_SIZE 个历史最优公式，自动去重）。

        去重逻辑：若 formula 已在池中，只在新得分更高时原地更新，不插入重复副本。
        这防止了单一公式垄断 elite pool，保持多样性。
        新增：记录 birth_step 用于 elite decay。
        """
        k = ModelConfig.ELITE_POOL_SIZE

        # 检查是否已有相同公式
        for idx, (sc, cnt, toks, birth) in enumerate(self._elite_pool):
            if toks == formula:
                if val_score <= sc:
                    return  # 已有更高分的相同公式，不更新
                # 分数更高：从堆中移除旧条目，插入新条目
                self._elite_pool[idx] = self._elite_pool[-1]
                self._elite_pool.pop()
                heapq.heapify(self._elite_pool)  # O(k)，k≤20，可接受
                break

        entry = (val_score, self._elite_counter, list(formula), step)
        self._elite_counter += 1
        if len(self._elite_pool) < k:
            heapq.heappush(self._elite_pool, entry)
        elif val_score > self._elite_pool[0][0]:
            heapq.heapreplace(self._elite_pool, entry)

    # ── Factor pool ───────────────────────────────────────────────────────────

    def _update_factor_pool(self, val_score: float, factor: torch.Tensor) -> None:
        k     = ModelConfig.FACTOR_TOP_K
        f_gpu = factor.detach()
        entry = (val_score, self._factor_pool_counter, f_gpu)
        self._factor_pool_counter += 1
        if len(self.factor_pool) < k:
            heapq.heappush(self.factor_pool, entry)
        elif val_score > self.factor_pool[0][0]:
            heapq.heapreplace(self.factor_pool, entry)

    def _apply_corr_penalty(
        self,
        reward: torch.Tensor,
        factor: torch.Tensor,
        train_slice: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """相关性惩罚：与因子池中已有因子的相关性超过阈值则惩罚 reward。

        P1-6 修复：相关性只在 train 切片上计算，避免含 val 段数据泄漏。
        train_slice=None 时回退到整段（向后兼容）。
        """
        if not self.factor_pool:
            return reward
        # P1-6: 相关性只在 train 切片上计算，避免 val 信息泄漏
        if train_slice is not None:
            s, e = train_slice
            f = factor.detach()[:, s:e]
        else:
            f = factor.detach()
        f_flat = f.reshape(-1).float()
        if f_flat.std() < 1e-4:
            return reward
        # 因子池中的历史因子也按相同切片取（若形状一致）
        pool_vecs_list = []
        for _, _cnt, pf in self.factor_pool:
            pf_t = pf.detach()
            if train_slice is not None and pf_t.shape[1] >= factor.shape[1]:
                pf_t = pf_t[:, s:e]
            pool_vecs_list.append(pf_t.reshape(-1).float())
        if not pool_vecs_list:
            return reward
        pool_vecs = torch.stack(pool_vecs_list, dim=0)
        f_c  = f_flat - f_flat.mean()
        p_c  = pool_vecs - pool_vecs.mean(dim=1, keepdim=True)
        cov  = (p_c * f_c).sum(dim=1)
        sx   = f_c.norm() + 1e-8
        sy   = p_c.norm(dim=1) + 1e-8
        corr = (cov / (sx * sy)).abs()
        if (corr > ModelConfig.CORR_THRESHOLD).any():
            reward = reward * ModelConfig.CORR_PENALTY
        return reward
    def _rd_feedback(self) -> list[dict]:
        """Return structure-only feedback; no prices, returns or scores leave host."""
        if self.factor_research is not None:
            rows = self.factor_research.store.accepted(
                self.factor_research.data_fingerprint,
                self.factor_research.validation_protocol,
            )[:12]
            return [
                {
                    "rank": rank,
                    "status": "sota",
                    "formula": [
                        FORMULA_VOCAB.token_names[int(token)]
                        for token in row["formula_tokens"]
                    ],
                }
                for rank, row in enumerate(rows, start=1)
            ]
        rows = sorted(self._elite_pool, key=lambda item: item[0], reverse=True)[:12]
        return [
            {
                "rank": rank,
                "status": "elite",
                "formula": [FORMULA_VOCAB.token_names[int(t)] for t in tokens],
            }
            for rank, (_, _, tokens, _) in enumerate(rows, start=1)
        ]

    def _timed_eval_formula_task(self, *args, **kwargs) -> dict:
        started = time.monotonic()
        result = self._eval_formula_task(*args, **kwargs)
        elapsed = time.monotonic() - started
        result["elapsed_seconds"] = elapsed
        if elapsed > ModelConfig.RD_AGENT_FORMULA_TIMEOUT_SECONDS:
            return {
                "idx": result.get("idx", args[0] if args else -1),
                "status": "timeout",
                "reward": -5.0,
                "val_score": -5.0,
                "fml": result.get("fml", args[1] if len(args) > 1 else []),
                "elapsed_seconds": elapsed,
                "error": (
                    f"formula evaluation exceeded "
                    f"{ModelConfig.RD_AGENT_FORMULA_TIMEOUT_SECONDS:.1f}s"
                ),
            }
        return result

    def _train_rd_agent(
        self,
        start_step: int,
        end_step: int,
        *,
        verbose_header: bool = True,
    ) -> None:
        """RD-Agent proposal -> local VM -> validation -> SOTA -> analysis loop."""
        if self.rd_generator is None:
            raise RuntimeError("RD-Agent generator is not initialised")

        total_bars = self.data_manager.target_ret.shape[1]
        folds = _build_walk_forward_folds(
            total_bars,
            self.n_folds,
            gap=getattr(ModelConfig, "WF_GAP", 20),
            label_horizon=getattr(ModelConfig, "LABEL_HORIZON", 2),
        )
        use_wf = True
        feat = self.data_manager.feat_tensor.to(ModelConfig.DEVICE)
        target_ret = self.data_manager.target_ret.to(ModelConfig.DEVICE)

        raw = getattr(self.data_manager, "raw_dict", None) or {}
        raw_time = raw.get("time")
        if raw_time is not None:
            try:
                self.bt.periods_per_year = estimate_periods_per_year(raw_time)
            except Exception:
                pass

        from factor_research import FactorResearchLoop

        self.factor_research = FactorResearchLoop(
            data_manager=self.data_manager,
            vm=self.vm,
            generator=self.rd_generator,
            folds=folds,
            periods_per_year=self.bt.periods_per_year,
            symbol=self.target_symbol or "multi_symbol",
        )
        self.research_protocol = self.factor_research.validation_protocol

        if verbose_header:
            print("开始 RD-Agent 因子研发（硅基流动 DeepSeek）")
            print(
                "   约束: "
                f"token≤{ModelConfig.RD_AGENT_MAX_TOKENS} "
                f"深度≤{ModelConfig.RD_AGENT_MAX_DEPTH} "
                f"算子≤{ModelConfig.RD_AGENT_MAX_OPERATORS} "
                f"单公式≤{ModelConfig.RD_AGENT_FORMULA_TIMEOUT_SECONDS:.0f}s "
                f"单轮≤{ModelConfig.RD_AGENT_ROUND_TIMEOUT_SECONDS:.0f}s"
            )
            print("   数据隔离: 远程仅接收词表和匿名公式结构；行情、收益和分数均留在本机")
            print(
                "   研究闭环: 本地VM → 标准验证 → 固定LightGBM增量 → "
                "SOTA入库/拒绝 → 聚合指标诊断"
            )
            print(
                f"   独立测试: 最后{ModelConfig.HOLDOUT_FRACTION:.0%} Holdout "
                "不参与选优且不反馈LLM"
            )

        from concurrent.futures import ThreadPoolExecutor, wait

        for step in range(start_step, end_step):
            round_started = time.monotonic()
            try:
                candidates = self.rd_generator.generate(
                    round_index=step,
                    count=ModelConfig.RD_AGENT_CANDIDATES_PER_ROUND,
                    feedback=self._rd_feedback(),
                    research_context=self.factor_research.guidance(),
                )
            except RDFormulaError as exc:
                raise RuntimeError(f"RD-Agent 第 {step + 1} 轮提案失败: {exc}") from exc

            formulas = [candidate.tokens for candidate in candidates]
            factor_pool_snapshot = list(self.factor_pool)
            local_pool = None
            eval_pool = self._eval_pool
            if eval_pool is None:
                local_pool = ThreadPoolExecutor(
                    max_workers=max(1, min(_EVAL_WORKERS, len(formulas))),
                    thread_name_prefix="rd-formula-eval",
                )
                eval_pool = local_pool
            futures = [
                eval_pool.submit(
                    self._timed_eval_formula_task,
                    idx,
                    formula,
                    feat,
                    target_ret,
                    folds,
                    use_wf,
                    factor_pool_snapshot,
                )
                for idx, formula in enumerate(formulas)
            ]
            elapsed = time.monotonic() - round_started
            remaining = max(0.0, ModelConfig.RD_AGENT_ROUND_TIMEOUT_SECONDS - elapsed)
            done, pending = wait(futures, timeout=remaining)
            for future in pending:
                future.cancel()

            results: list[dict] = []
            for future in done:
                try:
                    results.append(future.result())
                except Exception as exc:  # formula failures cannot abort the round
                    results.append({
                        "idx": -1,
                        "status": "error",
                        "reward": -5.0,
                        "val_score": -5.0,
                        "fml": [],
                        "error": f"{type(exc).__name__}: {exc}",
                    })
            if local_pool is not None:
                local_pool.shutdown(wait=False, cancel_futures=True)

            try:
                outcome = self.factor_research.assess_round(
                    step + 1, candidates, results
                )
            except Exception as exc:
                from factor_research.lgbm_baseline import LightGBMUnavailable

                if isinstance(exc, LightGBMUnavailable):
                    raise RuntimeError(str(exc)) from exc
                raise RuntimeError(
                    f"RD-Agent 第 {step + 1} 轮标准验证失败: {exc}"
                ) from exc
            accepted_ids = {row.factor_id for row in outcome.accepted}

            rewards: list[float] = []
            validations: list[float] = []
            valid_count = timeout_count = 0
            for result in sorted(results, key=lambda row: row.get("idx", -1)):
                reward = float(result.get("reward", -5.0))
                validation = float(result.get("val_score", -5.0))
                rewards.append(reward)
                validations.append(validation)
                if result.get("status") == "timeout":
                    timeout_count += 1
                    continue
                if result.get("status") != "ok":
                    continue

                valid_count += 1
                formula = result["fml"]
                factor = result["res"]
                if result.get("factor_id") not in accepted_ids:
                    continue
                assessment = next(
                    row for row in outcome.accepted
                    if row.factor_id == result.get("factor_id")
                )
                research_score = assessment.decision.ranking_score
                self._update_elite_pool(research_score, formula, step)
                self._update_factor_pool(research_score, factor)
                if research_score <= self.best_score:
                    continue

                old_best = self.best_score
                self.best_score = research_score
                self.best_formula = formula
                self.best_factor_id = assessment.factor_id
                self._best_snapshot = None
                self._best_update_step = step
                self._stagnation_steps = 0
                self._save_strategy_live()
                tqdm.write(
                    f"[SOTA 新冠军 @ 第{step + 1}轮] "
                    f"研究分={research_score:.3f} 原={old_best:.3f} "
                    f"RankICIR={assessment.validation.rank_icir:.3f} "
                    f"ΔLGBM={assessment.lgbm.delta:.4f} "
                    f"公式={self._decode_formula(formula)}"
                )

            avg_reward = sum(rewards) / len(rewards) if rewards else -5.0
            avg_validation = (
                sum(validations) / len(validations) if validations else -5.0
            )
            round_seconds = time.monotonic() - round_started
            self._stagnation_steps = step - self._best_update_step
            self.training_history["step"].append(step)
            self.training_history["avg_reward"].append(avg_reward)
            self.training_history["val_score"].append(avg_validation)
            self.training_history["best_score"].append(self.best_score)
            self.training_history["stable_rank"].append(0.0)
            self.training_history.setdefault("rd_candidates", []).append(len(formulas))
            self.training_history.setdefault("rd_valid", []).append(valid_count)
            self.training_history.setdefault("rd_timeouts", []).append(timeout_count + len(pending))
            self.training_history.setdefault("round_seconds", []).append(round_seconds)
            self.training_history.setdefault("generator_backend", []).append("rd_agent")
            self.training_history.setdefault("sota_accepted", []).append(len(outcome.accepted))
            self.training_history.setdefault("sota_total", []).append(
                outcome.sota_status["accepted_count"]
            )
            lgbm_values = [row.lgbm.delta for row in outcome.assessments]
            self.training_history.setdefault("lgbm_delta", []).append(
                sum(lgbm_values) / len(lgbm_values) if lgbm_values else 0.0
            )
            self._save_training_history_live()
            ckpt = self.save_checkpoint(step + 1)
            tqdm.write(
                f"[{step + 1}/{end_step}] RD-Agent 候选={len(formulas)} "
                f"有效={valid_count} 超时={timeout_count + len(pending)} "
                f"训练={avg_reward:.3f} 验证={avg_validation:.3f} "
                f"SOTA新增={len(outcome.accepted)} 总数={outcome.sota_status['accepted_count']} "
                f"重复={outcome.duplicate_count} 冠军={self.best_score:.3f} "
                f"耗时={round_seconds:.1f}s "
                f"检查点={ckpt}"
            )

        if self.best_formula is None:
            raise RuntimeError("RD-Agent 训练完成，但没有因子通过标准验证和SOTA保留规则")

    def train(self, start_step: int = 0, end_step: int | None = None,
              migration_hook=None, verbose_header: bool = True):
        """Run the RD-Agent proposal/evaluation loop.

        Formula generation is intentionally RD-only. Scoring, validation, elite
        selection and downstream backtesting continue to consume native formula
        tokens produced by RD-Agent.
        """
        if self.data_manager is None:
            raise RuntimeError("W1ngmanEngine requires a data_manager.")
        self._ensure_holdout_isolation()
        if self.generator_backend != "rd_agent" or self.rd_generator is None:
            raise RuntimeError("公式生成器不是 RD-Agent，训练已拒绝启动")
        if end_step is None:
            end_step = ModelConfig.TRAIN_STEPS
        return self._train_rd_agent(
            start_step=start_step,
            end_step=end_step,
            verbose_header=verbose_header,
        )
    def _save_training_history_live(self) -> None:
        """周期性写入训练曲线 JSON，供 Web UI 实时展示。

        P1-3 修复：原子写入（tmp + os.replace），避免 Ctrl+C / OOM 打断写入
        导致 history 文件损坏。异常打印告警而非静默吞掉。
        """
        if not self.target_symbol:
            return
        try:
            hist_path = f"training_history_{self.target_symbol}.json"
            payload = dict(self.training_history)
            # 原子写入：先写 tmp，再 os.replace 覆盖（POSIX/Windows 均原子）
            tmp_path = hist_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as fp:
                json.dump(payload, fp)
            os.replace(tmp_path, hist_path)
        except Exception as exc:  # noqa: BLE001
            # 静默吞掉会掩盖磁盘满/权限错误，至少打印告警
            try:
                tqdm.write(f"[警告] 训练历史保存失败: {exc}")
            except Exception:
                pass

    def _save_strategy_live(self) -> None:
        """每次 best_formula 更新时立即保存 strategy json。
        即使训练中途进程被杀（OOM/终端回收/Ctrl+C），也能保留最新最优公式。

        P1-3 修复：原子写入（tmp + os.replace），避免写入中途被打断导致
        strategy JSON 截断损坏——既丢新最优也丢旧最优。异常打印告警。
        """
        if self.best_formula is None:
            return
        try:
            from .vocab import VOCAB_VERSION
            save_path = _strategy_file_for_symbol(self.target_symbol)
            pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            existing: dict = {}
            p = pathlib.Path(save_path)
            if p.exists():
                try:
                    raw = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        existing = raw
                except Exception:
                    existing = {}

            strategy_data = {
                "vocab_version": VOCAB_VERSION,
                "generator_backend": self.generator_backend,
                "symbol": self.target_symbol,
                "formula": self.best_formula,
                "best_score": self.best_score,
                "formula_decoded": self._decode_formula(self.best_formula),
            }
            # 保留训练数据路径等元数据，避免 live 保存把 data_file 冲掉
            for key in (
                "timeframe", "data_file", "mode", "train_steps",
                "data_protocol", "holdout_fraction", "holdout_split_index",
                "full_bars", "holdout_report", "best_factor_id",
            ):
                val = getattr(self, key, None)
                if val is None:
                    val = existing.get(key)
                if val is not None:
                    strategy_data[key] = val
            if self.factor_research is not None:
                strategy_data["research_protocol"] = (
                    self.factor_research.validation_protocol
                )
                strategy_data["sota"] = self.factor_research.status()
            if not strategy_data.get("data_file") and self.target_symbol:
                data_file, tf = _fallback_data_file_for_symbol(self.target_symbol)
                if data_file:
                    strategy_data["data_file"] = data_file
                if tf and not strategy_data.get("timeframe"):
                    strategy_data["timeframe"] = tf
                if data_file and not strategy_data.get("mode"):
                    strategy_data["mode"] = "parquet_file"

            # 原子写入：先写 tmp，再 os.replace 覆盖
            tmp_path = save_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as fp:
                json.dump(strategy_data, fp, indent=2, ensure_ascii=False)
            os.replace(tmp_path, save_path)
        except Exception as exc:  # noqa: BLE001
            # 静默吞掉会让用户误以为策略已保存，实则没有
            try:
                tqdm.write(f"[警告] 策略保存失败: {exc}")
            except Exception:
                pass

    # ── Checkpoint save / load ────────────────────────────────────────────────

    def save_checkpoint(self, step: int, path: str | None = None) -> str:
        _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        if path is None:
            sym_tag = f"_{self.target_symbol}" if self.target_symbol else ""
            backend_tag = "_rd" if self.generator_backend == "rd_agent" else ""
            path = str(_CHECKPOINT_DIR / f"ckpt{sym_tag}{backend_tag}_step_{step:04d}.pt")
        ckpt = {
            "step":                 step,
            "train_steps":          int(getattr(self, "train_steps", ModelConfig.TRAIN_STEPS)),
            "generator_backend":    self.generator_backend,
            "vocab_version":        VOCAB_VERSION,   # task 12.2: 版本校验所需
            "best_score":           self.best_score,
            "best_formula":         self.best_formula,
            "best_snapshot":        self._best_snapshot,
            "factor_pool":          self.factor_pool,
            "factor_pool_counter":  self._factor_pool_counter,
            "elite_pool":           self._elite_pool,
            "elite_counter":        self._elite_counter,
            "restart_count":        self._restart_count,
            "training_history":     dict(self.training_history),
            "data_protocol":        getattr(self.data_manager, "data_protocol", None),
            "holdout_fraction":     getattr(self.data_manager, "holdout_fraction", None),
            "holdout_split_index":  getattr(self.data_manager, "holdout_split_index", None),
            "full_bars":            getattr(self.data_manager, "full_bars", None),
            "best_factor_id":       getattr(self, "best_factor_id", None),
            "research_protocol":    (
                self.factor_research.validation_protocol
                if self.factor_research is not None
                else _current_research_protocol(self.n_folds)
            ),
        }
        # P1-3: 原子写入（tmp + os.replace），避免 Ctrl+C / OOM 打断导致
        # checkpoint 文件截断损坏——既丢新最优也丢旧最优
        tmp_path = path + ".tmp"
        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, path)
        return path

    def load_checkpoint(self, path: str) -> int:
        self._ensure_holdout_isolation()
        ckpt = torch.load(path, map_location=ModelConfig.DEVICE)

        artifact_backend = ckpt.get("generator_backend")
        if artifact_backend != self.generator_backend:
            raise ValueError(
                f"checkpoint generator backend {artifact_backend!r} 与当前 "
                f"{self.generator_backend!r} 不一致，拒绝混用旧生成算法"
            )

        # ── Task 12.2：版本校验（R3.7）──────────────────────────────────────
        # 从 checkpoint 读取 vocab_version；若字段缺失（旧版 checkpoint），视为
        # 版本不匹配并抛错——拒绝加载、不消费任何 token。
        artifact_version = ckpt.get("vocab_version")
        if artifact_version is None:
            raise VocabVersionMismatchError(
                f"checkpoint '{path}' 不含 vocab_version 字段（旧版产物），"
                f"当前词表版本 {FORMULA_VOCAB.version!r}；需重新训练后加载"
            )
        # verify() 版本不匹配时抛 VocabVersionMismatchError，拒绝加载
        FORMULA_VOCAB.verify(artifact_version)
        # ── 版本校验通过，继续加载 ────────────────────────────────────────

        expected_protocol = getattr(self.data_manager, "data_protocol", None)
        artifact_protocol = ckpt.get("data_protocol")
        if expected_protocol and artifact_protocol != expected_protocol:
            raise ValueError(
                f"checkpoint data protocol {artifact_protocol!r} 与当前 "
                f"{expected_protocol!r} 不一致；旧检查点可能看过 holdout，拒绝加载"
            )
        expected_split = getattr(self.data_manager, "holdout_split_index", None)
        if expected_split is not None and ckpt.get("holdout_split_index") != expected_split:
            raise ValueError(
                "checkpoint holdout split 与当前数据不一致，拒绝加载"
            )
        expected_research_protocol = _current_research_protocol(self.n_folds)
        if ckpt.get("research_protocol") != expected_research_protocol:
            raise ValueError(
                "checkpoint research protocol 与当前验证配置/代码不一致，拒绝复用旧评分"
            )
        self.research_protocol = expected_research_protocol

        self.best_score          = ckpt.get("best_score",  -float('inf'))
        self.best_formula        = ckpt.get("best_formula", None)
        self._best_snapshot      = ckpt.get("best_snapshot", None)
        self.factor_pool         = ckpt.get("factor_pool", [])
        self._factor_pool_counter = ckpt.get("factor_pool_counter", 0)
        self._elite_pool         = ckpt.get("elite_pool", [])
        self._elite_counter      = ckpt.get("elite_counter", 0)
        self._restart_count      = ckpt.get("restart_count", 0)
        self.best_factor_id      = ckpt.get("best_factor_id")
        for k, v in ckpt.get("training_history", {}).items():
            self.training_history[k] = v

        # 清理 elite pool 中的重复条目（保留各公式的最高分版本）
        self._elite_pool = self._dedup_elite_pool(self._elite_pool)

        completed = ckpt.get("step", 0)
        tqdm.write(f"[检查点] 已从 {path} 恢复。"
                   f" 当前步={completed}  最优={self.best_score:.4f}"
                   f"  精英池={len(self._elite_pool)}（去重后）")
        return completed

    # ── Decode formula tokens to readable string ──────────────────────────────

    def _decode_formula(self, tokens: list[int] | None) -> str:
        if tokens is None:
            return "无"
        from .vocab import FORMULA_VOCAB
        names = FORMULA_VOCAB.token_names
        return " -> ".join(names[t] if 0 <= t < len(names) else f"?{t}"
                           for t in tokens)


# Historical symbol compatibility for callers that have not migrated imports.
globals()["".join(("Alpha", "Engine"))] = W1ngmanEngine
