# D21-v2 冻结（保留 D21-v1 证据）

状态：`CONFIRMED_FREEZE_20260901_R1`  
假设：`H2_5D_SLOW_RESIDUAL_BUFFER_V1`  
`development_adaptive`: true  
Holdout：未读  
D21-v1：保留 `SIMPLE_ENSEMBLE_V1+NAIVE_TOP20` 及其 `KEEP_1D_BASELINE`，产物不得覆盖  
执行结果（2026-09-01）：Stage 4R = `INSUFFICIENT_EVIDENCE`（OOF 慢速残差覆盖率 77.7% < 95%）。失败产物保留。不得据此改窗口；下一次必须 `D21-v3`。

本文件是用户确认的第二代第一阶段参考对象。它不替换 2026-08-25 的 D21-v1。v1 失败结论仍然有效。v2 只有自己通过 Stage 3R/4R 才产生 `HORIZON_FEASIBLE_5D_V2`。通过也不等于生产模型有效，不得启动阶段5或读取 Holdout。

## 研究对象

```text
hypothesis_id:        H2_5D_SLOW_RESIDUAL_BUFFER_V1
signal_version:       SLOW_RESIDUAL_ENSEMBLE_V1
portfolio_version:    BUFFERED_TOP20_5D_V1
target_horizon:       5
development_adaptive: true
holdout_read:         false
```

只研究 5 日，不同时研究 3 日。选择 5 日受 D21-v1 Development 结果启发，因此必须标记 `development_adaptive=true`。

相对门槛的对照是实验 A（同一 5 日、`SIMPLE_ENSEMBLE_V1` + `NAIVE_TOP20_5D_V1`），不是 D21-v1 的 1 日每日重构。

## D18-v2 绝对经济门槛

成本 ×1、五个 5 日相位中位数：

- 候选 D 自身净 Sharpe **必须 > 0**
- 候选 D 自身扣费年化 **必须 > 0**

相对 A 仍沿用 D18 的 +0.15 Sharpe 与 +2pct 年化，以及 Bootstrap CI 下界 > 0。不得因为“比旧 1 日少亏”就宣布通过。

## 行业残差

```text
r(i,t) = log(CloseTR(i,t) / CloseTR(i,t-1))
r_industry_LOO(i,t) = mean(r(j,t)) for j != i and industry(j,t) = industry(i,t)
epsilon(i,t) = r(i,t) - r_industry_LOO(i,t)
```

- 行业为 t 日点时申万一级；Leave-One-Out
- 同行业除目标外有效成员 < 5：残差缺失
- 行业缺失不得用当前行业回填；收益/停牌/缺行情不得填 0
- 残差在研究交易日历上滚动（不是成员区间日）
- 本快照没有全市场行业行情时，LOO 宇宙为当日点时沪深300成员中同行有效股票

## 三个信号

| 信号 | 定义 | 有效样本 |
|---|---|---|
| `IDIO_LOW_VOL_60` | `-std(epsilon[t-59:t], ddof=0)` | ≥54/60 |
| `RES_MOM_120_20` | `(100/n_valid)*sum(epsilon[t-119:t-20])` | ≥90 个有效残差；跳过最近 20 日 |
| `RES_TREND_EFF_60_5` | `sum(eps[t-59:t-5]) / (sum(abs(eps[t-59:t-5]))+1e-12)` | ≥50 个有效残差；跳过最近 5 日 |

分数越高越优先。不足则缺失。不是把 v1 的 `LOW_VOL_20` 原样留下。

## 截面集成

当日点时成员：1%/99% 截尾 → 平均秩百分位 → 三个信号全部有效才等权平均。缺一不可重平均。不学习权重。OOF 成员日有效覆盖率 < 95% → `INSUFFICIENT_EVIDENCE`。

## 5 日排名缓冲

五个相位 `0/1/2/3/4` 独立运行。

- 新进入：`entry_rank <= 20` 且可买、流动性门槛同 D13
- 继续持有：已持仓且 `current_rank <= 40` 且仍是成员且信号有效
- 退出：`rank > 40` 或信号无效或不再是成员；卖出仍受执行约束，阻塞则 `exit_pending`
- 补充：只从当日 Top20 未持有股票按排名补到 20；否则现金；不用第 21 名以后替补
- 每股目标 5%，最多 20 只，总目标 ≤ 100%
- 实际持仓因无法卖出超过 20 只时不开新仓
- Top40 = 固定 `2×K`，看结果后不得改成 Top30/50

## 必须运行的四组消融

| 实验 | 信号 | 执行 |
|---|---|---|
| A | `SIMPLE_ENSEMBLE_V1` | `NAIVE_TOP20_5D_V1` |
| B | `SLOW_RESIDUAL_ENSEMBLE_V1` | `NAIVE_TOP20_5D_V1` |
| C | `SIMPLE_ENSEMBLE_V1` | `BUFFERED_TOP20_5D_V1` |
| D | `SLOW_RESIDUAL_ENSEMBLE_V1` | `BUFFERED_TOP20_5D_V1` |

最终门禁候选是 D。A–D 全部进入报告和 Holm 家族。`B-A` 新特征，`C-A` 缓冲，`D-C` 同缓冲下新特征，`D-A` 最终净改善。

产物目录（不得覆盖 v1）：

```text
experiments/stage3r_d21_v2_h2_5d
experiments/stage4r_d21_v2_h2_5d
```

## 禁止

- 只保留三个新信号中最好的一个
- 改窗口、改 Top20/Top40、改成行业内排名、学习权重
- 试多个退出阈值后挑最好
- RD-Agent 补第四个信号
- 用 Holdout 选特征
- 覆盖 `experiments/stage4_hs300_v2_execution_ledger`
- 把 v2 Development 通过解释成生产有效
- 失败后原地改参数；下一次必须 `D21-v3` 和新假设记录
