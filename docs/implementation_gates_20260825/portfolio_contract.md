# 组合合同（拟冻结 v1）

状态：`CONFIRMED_FREEZE_20260825`  
实施阶段：阶段6；阶段3仅实现 Top20 等权研究参考组合。

2026-09-16工程增补：独立离线评分／白名单／配置模拟已实现，见[运行与API文档](offline_portfolio_workflow_20260916.md)。当前仅合成历史回放，采用 `CAPPED_EQUAL_WEIGHT`受约束等权；没有实现本文第2节的预期收益与协方差优化，也不构成阶段6或正式生产门禁通过。历史参考组合、原研究结论和源评分包合同不变。

## 1. 输入职责

| 输入 | 只负责 |
|---|---|
| 个股 Alpha | 白名单/参考池内优先级、预期收益和置信度。 |
| 市场状态 | 总风险预算、总仓位上限、现金下限。 |
| 行业状态 | 行业预算与行业集中约束。 |
| 风险模型 | 波动、相关性和组合风险。 |
| 当前持仓 | 换手、成本、延迟退出和行为差异。 |
| 执行掩码 | t+1 是否可买/可卖；不得反向修改 t 日信号。 |

同一市场/行业风险不能同时加进个股 Alpha 和预算。

## 2. 优化目标

```text
maximize
expected_return · weight
- risk_aversion × portfolio_risk
- turnover_penalty × turnover
- direction_aware_transaction_cost
- industry_concentration_penalty
```

模型分数不能直接当权重；`tanh(factor)` 不能进入正式组合路径。

## 3. 硬约束

1. 用户白名单外权重严格为0。
2. 非信号日点时沪深300成员不得新增仓位。
3. A股普通模式权重≥0。
4. `sum(stock_weights)+cash_weight=1`，现金≥0。
5. 信号不足时允许100%现金。
6. 单股目标权重≤5%（D22已确认）。
7. 单一申万一级行业目标权重≤30%（D22已确认）。
8. 总股票权重≤市场风险预算。
9. 停牌不得成交；一字涨停不得买，一字跌停不得卖。
10. 流动性不达标不得新开仓。
11. 调出成员不得新增，已有仓位按退出状态机处理。
12. 退市损失和延迟退出不得从净值删除。
13. 约束不可行不得生成随机或负现金权重。

## 4. 研究参考组合

阶段3没有人工白名单。每个调仓日从点时沪深300有效成员中按 `SIMPLE_ENSEMBLE_V1` 取 Top20，每只5%。不可买的目标资金留现金，不用第21名临时替补。该规则保证跨1/3/5日可比，不能解释为最终用户配置器。

## 5. 用户白名单

### 2026-09-15产品增补：沪深300全量列表后由用户选择

用户最新确认输出当期沪深300全部300只股票：最终公式确定后，在点时沪深300全池上计算“最终公式给每只股票的当期得分”，完整输出并按有效得分降序排列，用户从完整列表中选择股票，组合层再对所选股票配置权重。日线模式下使用同一信号日完整收盘数据可用时点的得分；IC / RankIC用于研究质量评价，不作为个股排序字段。

有效有限得分按降序排列，同分按规范股票代码升序；无有效得分的成员仍保留在列表末尾，按代码排序，得分及得分排名为空并报告原因。不能删除无效得分成员、填0或使用过期分数；成员数据不完整时报告数据错误。完整列表保存独立ID/版本及当时的排名、得分/状态和公式/数据版本，用户白名单必须关联其所见快照，所选股票必须是该快照的子集。

阶段9白名单保存：ID/版本、候选快照ID/版本、创建时间、适用交易日、股票列表、备注、数据/股票池/行业/策略版本。白名单只限制可配置股票，不改变完整沪深300全池排名和完整行业参照。用户未选择时不默认配置；不能将全部300只自动持有或向白名单补入未选股票。无有效得分的所选股票不得据此生成正目标权重，须报告证据不足。

该变更适用于产品完整列表，第4节的Top20研究参考组合保留原口径。既有单股5%、行业30%、总风险预算及现金约束继续适用；展示300只股票不改变最终持仓数和权重规则。候选刷新、白名单更新与实际持仓分开记录，执行受阻的旧仓仍按第6节处理。完整产品需求见[产品待办](../03_产品待办.md)。

当前离线实现将完整评分快照、白名单各版本和方案另存 `portfolio_demo_store/portfolio.sqlite3`。名单更新以 `expected_version`检查冲突，方案固定绑定所见快照与名单版本。300只合成包保留无效分数成员，16只样例明确标记为小样例；Development包要求每期完整300成员，但当前尚未接入真实数据。

离线权重仅分配给用户所选且评分有效的股票；未知行业统一计入同一30%上限，现金含预留费用。缺持仓时只给目标、不估算调仓；执行信息未知时不假定已成交。明确不可卖的旧仓继续保留并消耗总预算和行业额度，新买入受剩余现金、费用及风险空间约束。该单次条件模拟没有实现连续前向执行状态机。评分来源 `model_prediction`仅作排序，不代表单公式收益率、预期收益或权重。

当前人工白名单不得回填历史作为无偏结论；可信结果来自阶段10前向影子运行。

## 6. 信号与执行状态机

```text
t日 available_at 后生成目标
→ 保存 SIGNAL_GENERATED
→ t+1开盘检查 can_buy_open/can_sell_open
→ 可执行：模拟成交并记实际成本
→ 不可执行：TARGET_VALID_BUT_EXECUTION_BLOCKED
→ 后续每日重试退出/调整，不改写原信号
```

调出指数：

```text
EXIT_EFFECTIVE
→ target=0, no_new_buy=true
→ 可卖则下一开盘退出
→ 不可卖则UNIVERSE_EXIT_PENDING
→ 仓位/PnL/风险持续
→ EXIT_COMPLETED 或 DELISTING_DISPOSITION
```

## 7. 确定性回退

| level | 行为 | 状态 |
|---:|---|---|
| 0 | 原优化器可行解 | `PORTFOLIO_GATE_PASSED` |
| 1 | 降低总风险预算后重解 | `FALLBACK_PORTFOLIO` |
| 2 | 合法正信号股票风险平价 | `FALLBACK_PORTFOLIO` |
| 3 | 合法正信号股票等权 | `FALLBACK_PORTFOLIO` |
| 4 | 全部现金 | `NO_TRADE` 或 `CONSTRAINT_INFEASIBLE` |

每次保存：`fallback_level, fallback_reason, original_constraints, conflicting_constraints, final_weights, cash_weight`。给定相同输入必须输出完全相同结果。

## 8. 成本

换手按目标与当前可执行权重差计算，买卖分向扣费。部分执行只对实际成交部分收费；未成交权重保持原仓并继续计风险。没有账户规模时只按比例成本，不输出股数和最低佣金。

## 9. 输出字段

组合级：`as_of, execution_window, horizon, status, reason_codes, market_risk_budget, expected_volatility, expected_turnover, expected_cost, cash_weight, max_industry_weight, fallback_level`。

股票级：`code, industry, universe_rank, industry_rank, alpha, confidence, current_weight, target_weight, delta_weight, action, tradability, block_reason, exit_state`。

若缺当前持仓：只给目标权重，不给“增持/减持/卖出”。若缺账户资金：不给股数。

## 10. 组合门禁状态

只允许：`PORTFOLIO_GATE_PASSED, NO_TRADE, CONSTRAINT_INFEASIBLE, EXECUTION_BLOCKED, UNIVERSE_EXIT_PENDING, FALLBACK_PORTFOLIO, PORTFOLIO_GATE_FAILED`。研究模型未验证时，即使约束测试通过，也不得输出正式 `OK` 建议。
