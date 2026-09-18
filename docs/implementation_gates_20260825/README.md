# quant_w1ngman 分阶段实施门禁包

## 2026-09-15：联合研究S2（新增）

用户已批准[阶段4/5联合研究S2](joint_research_s2_20260915.md)及工程修改：新路线撤销旧阶段4/原三个信号先通过的前置，内层联合筛选/拟合、外层评价完整流程；低IC不自动淘汰。具体候选、日期、输出方式、总预算及组合风险合同仍待冻结，当前不是正式实验GO。

D21-v3已完成，4R结果为 `D21_V3_FAILED`；旧v1/v2/v3结果均保留，以下为各次历史登记。S2首版外层预测工程不自动完成交易经济验证，不启用动态公式/RD-Agent或写SOTA；系统仍为 `MODEL_NOT_VALIDATED`，Holdout和生产继续关闭。

S2入口：[联合计划](joint_research_s2_20260915.md)、[未冻结配置模板](joint_research_s2_plan.example.json)。模块为 `factor_research/joint_protocol.py` 与 `factor_research/joint_runner.py`；仓库根目录运行 `D:/anaconda/python.exe scripts/check_joint_research_plan.py docs/implementation_gates_20260825/joint_research_s2_plan.example.json` 仅做只读检查，不启动搜索。示例5×480=2400仅覆盖外层搜索，不含尚未实现的最终重选；模板预期返回 `PLAN_CHECK_BLOCKED`。

审计日期：2026-09-01（Asia/Shanghai）  
状态：`KEEP_1D_BASELINE`  
系统状态：`MODEL_NOT_VALIDATED`  
数据研究状态：`DATA_READY_FOR_DEVELOPMENT`  
统计门禁：`KEEP_1D_BASELINE`

阶段2推荐方案已于 2026-08-25 由用户确认。正式研究快照为 `D:/Hulucoding/AmAzing_Data/research_snapshots/csi300_2014_present_v2/`（研究起点 2014-01-02）。2026-08-31 已按执行账本规则重冻：成员面板仍按当日成分切片；成交与估值使用 `standardized/execution_bars.parquet` / `execution_status.parquet`；成分进出审计在 `audit/membership_spell_audit.parquet`。面板数据门禁为 `DATA_GATE_PASSED`（行业覆盖率 98.22%，未映射 WARNING 13,845 行）。Holdout 价格在 `sealed/holdout/`，读取需要一次性 `sealed/holdout.capability`，并写入 `sealed/holdout.access.log`。这不是独立 OS 用户、HSM 或加密保险库。不得把可加载张量当成正式策略结论。v1 raw 只读、未重采。

现行 Development OOF 为 `experiments/stage3_hs300_v2_execution_ledger` 与 `experiments/stage4_hs300_v2_execution_ledger`（协议 `w1ngman_stage4_feasibility_oof_only_v2`）。3 日和 5 日仍未通过预注册 GO。2026-08-28 成员切片回测归档在 `experiments/stage3_hs300_v2` / `experiments/stage4_hs300_v2`，不得当作现行证据。2026-09-01 追加 D21-v2（`d21_v2_freeze.md`）：保留 v1 结论，独立跑 Stage 3R/4R，Holdout 仍封存。系统保持 `MODEL_NOT_VALIDATED`。不得启动 RD-Agent，不得进入阶段5。

## 当前四类门禁

| 门禁 | 状态 | 结论 |
|---|---|---|
| 数据门禁 | `PASSED` | `HS300PanelDataManager.load` 在 v2 上通过。硬验收：每日 300、成员例外 0、OpenTR 前缀不变。WARNING：13,845 个 Development 成员日无申万一级（覆盖率 98.22% ≥ 95%）。Holdout 价格只读封存，一次性 capability 未消耗。 |
| 统计门禁 | `KEEP_1D_BASELINE` | 执行账本重冻后阶段4再次完成配对 Bootstrap、Holm 与预注册 GO。3 日和 5 日仍未通过；净增量主要来自换手下降而非毛 Alpha。 |
| 组合门禁 | `REFERENCE_ONLY` | Top20等权长仓参考回测已实现；正式白名单组合和优化器属于阶段6。 |
| 产品门禁 | `PARTIAL` | 旧训练/回测网页可导入。`/api/v2` 状态接口与拒绝建议骨架已实现（空仓位、现金 100%）。正式白名单、可交易目标权重与阶段9产品化仍关闭。 |

产品门禁局部可用不改变系统状态；正式建议必须继续返回 `MODEL_NOT_VALIDATED`。

## 文件索引

阶段 0：

- `project_map.md`
- `asset_inventory.json`
- `current_feature_matrix.md`
- `compatibility_inventory.md`
- `audit_scope.md`

阶段 1：

- `evidence_audit.md`
- `signal_trace.json`
- `leakage_findings.json`
- `baseline_reproduction.json`
- `baseline_environment.lock`

阶段 2：

- `frozen_design.md`
- `decision_table.md`
- `data_dictionary.md`
- `validation_protocol.md`
- `portfolio_contract.md`
- `api_contract.md`
- `experiment_registry_schema.md`

阶段 3：

- `confirmation_record.json`
- `stage3_implementation.md`
- `stage3_test_report.json`

阶段 4：

- `stage4_implementation.md`
- `stage4_test_report.json`

KEEP_1D 收口（不是阶段5）：

- `keep_1d_closure.md`

D21-v2（保留 v1 证据的新假设，不是阶段5）：

- `d21_v2_freeze.md`

## 下一门禁

KEEP_1D 工程收口已完成：系统状态 API 报告 `MODEL_NOT_VALIDATED`，建议接口失败关闭。D21-v1 的 `KEEP_1D_BASELINE` 不得覆盖。D21-v2 Stage 3R/4R 已跑完：OOF 慢速残差覆盖率 77.7% < 95%，结论 `INSUFFICIENT_EVIDENCE`，失败产物保留在 `experiments/stage3r_d21_v2_h2_5d` 与 `experiments/stage4r_d21_v2_h2_5d`。不得用该结果改窗口或改成 D21-v2 原地补丁；下一次修改必须是 `D21-v3`。阶段5、RD-Agent 与 Holdout 仍关闭。
