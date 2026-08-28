# 阶段4 Development OOF 可行性门禁

状态：`KEEP_1D_BASELINE`  
系统状态：`MODEL_NOT_VALIDATED`  
统计门禁：`KEEP_1D_BASELINE`  
RD-Agent：不允许  
Holdout：未读取、未计算表现  
产品待办：保持关闭

本阶段回答的唯一预注册问题是：在冻结的点时沪深300 Top-20 等权长仓参考组合、同一标签、成本、切分和 `SIMPLE_ENSEMBLE_V1` 下，3 日或 5 日是否相对 1 日具有稳定、扣费后仍有经济意义的 Development OOF 增量。

结论：**没有。** 软件按协议完成了门禁；研究假设未通过。这不是软件故障。

## 已实现

- `research_stage4/runner.py`：协议 `w1ngman_stage4_feasibility_oof_only_v2`。只消费阶段3 Development 产物；缺失四个必需文件时失败关闭，不自动重跑阶段3。
- OOF 范围：特征/标签先按阶段3验证折日期裁剪，再做稳健性。全量 Development 特征不得当作 OOF。
- 配对移动块 Bootstrap（块长 20 日、2000 次、种子 `20260812`）+ Holm 家族 5%。
- 预注册 GO 阈值未随结果修改。
- 相位一致性：中位数相对 1 日的符号为 0 时，不再把全部相位计为同意。
- 成本披露：OOF 日频 `gross_return`/`net_return` 拆成毛 Alpha、成本节约、净改善；该拆分不进入 GO。
- `run_stage4_feasibility.py`：命令行入口。

第一期比较树只做周期，再做相位稳健性。未实现、也未纳入 GO：

- staggered 子组合
- 市场风险开关
- 行业预算或三层规则
- FormulaValidationScorer / RD-Agent
- 组合优化器、正式建议 API、网页产品化

## 复现命令

```powershell
cd C:\Users\Administrator\Documents\Codex\2026-07-31\yue-2\quant_w1ngman
D:\anaconda\python.exe run_stage4_feasibility.py `
  --snapshot "D:\Hulucoding\AmAzing_Data\research_snapshots\csi300_2014_present_v2" `
  --stage3 "experiments\stage3_hs300_v2" `
  --output "experiments\stage4_hs300_v2"
```

输入必须已存在：

```text
experiments/stage3_hs300_v2/stage3_report.json
experiments/stage3_hs300_v2/development_oof_portfolio_daily.parquet
experiments/stage3_hs300_v2/development_features.parquet
experiments/stage3_hs300_v2/development_labels.parquet
```

正式输出：`experiments/stage4_hs300_v2/stage4_report.json`  
旧协议 `w1ngman_stage4_feasibility_v1` 备份为 `stage4_report_protocol_v1.json`，不得当作本轮结论。

## 冻结口径（未改）

| 项 | 值 |
|---|---|
| 快照 | `csi300_2014_present_v2` |
| 配置哈希 | `5a25efafe5c095009383e1d510c8bca2e47990afb249de2815b2e089075ee7cd` |
| Development | 2014-01-02 → 2024-08-23，2591 日 |
| OOF 日期 | 2018-04-03 → 2024-08-22，1525 日（5 个 expanding 验证折） |
| Holdout | 2024-08-26 起 483 日；本阶段未读价格或表现 |
| 参考组合 | Top 20 等权、每只 5%、只做多加现金 |
| 成本 ×1 | 买 7.6bp / 卖 12.6bp |
| 主指标 | 全相位中位数净 Sharpe |
| GO | 净 Sharpe +0.15 且扣费年化 +2pct；CI 下界 > 0；Holm 5%；3 日相位 ≥2/3；5 日相位 ≥3/5；折 ≥3/5；信号 ≥2/3；回撤 ≤1 日的 1.2 倍 |

OOF 裁剪：特征使用 457,500 行，排除非 OOF 319,800 行；标签使用 4,117,500 行，排除 2,878,200 行。

## 主结果（Development OOF，成本 ×1）

| 周期 | 全相位中位净 Sharpe | 中位扣费年化 | 中位最大回撤 | 相对 1 日 ΔSharpe | 相对 1 日 Δ年化 |
|---|---:|---:|---:|---:|---:|
| 1 日 | -0.655 | -11.0% | 55.4% | — | — |
| 3 日 | -0.383 | -7.1% | 41.6% | +0.272 | +3.88pct |
| 5 日 | -0.174 | -4.1% | 34.2% | +0.481 | +6.94pct |

全部 3 个 3 日相位、全部 5 个 5 日相位的净 Sharpe 都高于 1 日，且全部仍为负。不准只报最好相位：3 日最差相位 Sharpe = -0.385；5 日最差 = -0.293。

配对 Bootstrap 与 Holm 均拒绝“无改善”：3 日 CI `[0.093, 0.544]`，5 日 CI `[0.199, 0.759]`，Holm p ≈ 0.002。统计上的“相对 1 日更好”成立，但预注册辅助门槛没有同时满足，因此不能 GO。

## 成本披露（不进入 GO）

OOF 日频毛/净拆分满足 `净改善 = 毛 Alpha + 成本节约`：

| 周期 | 毛年化 | 成本拖累年化 | 净年化 | 相对 1 日毛 Alpha | 相对 1 日成本节约 | 相对 1 日净年化 | 日均换手 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 日 | +6.2% | 17.2pct | -11.0% | — | — | — | 73.3% |
| 3 日 | +2.3% | 9.4pct | -7.1% | **-3.93pct** | +7.81pct | +3.88pct | 40.0% |
| 5 日 | +3.0% | 7.1pct | -4.1% | **-3.18pct** | +10.11pct | +6.94pct | 29.2% |

阶段3折内成本敏感性（中位净 Sharpe）：

| 周期 | ×0 | ×1 | ×1.5 |
|---|---:|---:|---:|
| 3 日 | +0.308 | -0.268 | -0.630 |
| 5 日 | +0.278 | -0.211 | -0.467 |

扣费后更“不那么差”的是 5 日，但 3 日和 5 日在成本 ×1 与 ×1.5 下净 Sharpe 均为负。相对 1 日的净增量主要来自换手下降，而不是更强的毛 Alpha。毛 Alpha 为负，因此不能把 3/5 日解释为可部署的预测优势。

## GO 检查

### 3 日：未通过

通过：Sharpe 改善、年化改善、Bootstrap CI、Holm、相位方向 3/3、最差相位下限、回撤、集中度年份/行业。

未通过：

- 折方向 1/5（门槛 3）
- 信号方向 1/3（门槛 2）：仅 `LOW_VOL_20` 的 RankIC 高于 1 日；`MOM_20`、`REV_5` 更差

### 5 日：未通过

通过：Sharpe 改善、年化改善、Bootstrap CI、Holm、相位方向 5/5、最差相位下限、回撤。

未通过：

- 折方向 1/5
- 信号方向 1/3（同样只有 `LOW_VOL_20`）
- 集中度：剔除贡献最大 5% 股票后 RankIC 增量不再与总体同号；行业 Leave-One-Out 下多数行业的 RankIC 增量翻号

v2 相对旧 v1 的实质变化来自“只用 OOF 日期做稳健性”，不是改阈值：3 日折同意从 2 降到 1；5 日集中度从通过变为不通过。主指标点估计与 Bootstrap 与 v1 一致。

## 预注册问题的答案

| 问题 | 答案 |
|---|---|
| 3 日是否具有稳定样本外增量 | 否。净 Sharpe 仍为负；折与信号不一致。 |
| 5 日是否具有稳定样本外增量 | 否。同上，且个股集中度检查失败。 |
| 哪个周期扣费后更合理 | 5 日净损失小于 3 日、3 日小于 1 日，但三者扣费后均不可用。 |
| 结果是否依赖特定调仓相位 | 方向上不依赖；幅度上 5 日相位 Sharpe 从 -0.29 到 -0.08。已报告全部相位。 |
| 行业特征是否有边际贡献 | 第一期 GO 未启用行业层；5 日 RankIC 增量在多数行业 Leave-One-Out 下不稳健。 |
| 市场状态是否降低风险 | 第一期未启用市场层，不作为本门禁证据。 |
| 三层简单规则是否优于单层基线 | 未比较。按冻结比较树，周期未通过则停止增加复杂度。 |
| 改善是否具有经济意义 | 否。净改善来自降低换手；毛 Alpha 为负；×1/×1.5 净 Sharpe 为负。 |
| 改善是否只来自某一行业或某一时期 | 净 Sharpe 的年份剔除方向一致；5 日 RankIC 增量对头部股票和多数行业敏感。 |

门禁映射：`KEEP_1D` / `KEEP_1D_BASELINE`。系统继续返回 `MODEL_NOT_VALIDATED`。第一期不使用 `REJECT_COMPLEXITY`（复杂规则尚未启用）。

## 四类结论必须分开

1. **软件功能**：阶段4门禁可复现，产物完整，Holdout 隔离有效。
2. **研究假设**：3 日/5 日相对 1 日的预注册可行性 **未通过**。
3. **是否允许生成建议**：否。不得输出具有确定性交易含义的建议。
4. **Holdout**：未使用。该区间仍保持封存，不得因本结果去打开它。
5. **前向影子**：不启动。产品待办保持关闭。

不得因为软件运行正常就声称模型有效。不得把 `MODEL_NOT_VALIDATED` 描述为程序失败。

## 下一步（需用户决定）

允许：

- 保留 1 日简单参考方案；
- 停止增加公式搜索、评分器、优化器和建议 API；
- 若用户明确提出**新的研究假设**（例如改成本口径、改参考组合、补数据版本），必须先改阶段2冻结表并重新预注册，不能在本结果上调 GO 阈值。

不允许在未改假设的前提下进入阶段5。

## 测试

```powershell
D:\anaconda\python.exe -m pytest tests/unit/test_stage4_feasibility.py -q `
  --basetemp "C:\Users\Administrator\Documents\Codex\2026-07-31\yue-2\quant_w1ngman\.pytest-tmp-stage4"
```

12 passed，约 181 秒。覆盖 Holm、Bootstrap 种子、GO 阈值、Holdout 泄漏失败关闭、OOF 折溯源、非 OOF 行剔除、缺失标签失败关闭、相位一致性和成本恒等式。
