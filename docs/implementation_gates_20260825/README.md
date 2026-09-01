# quant_w1ngman 分阶段实施门禁包

审计日期：2026-09-01（Asia/Shanghai）  
状态：`KEEP_1D_BASELINE`  
系统状态：`MODEL_NOT_VALIDATED`  
数据研究状态：`DATA_READY_FOR_DEVELOPMENT`  
统计门禁：`KEEP_1D_BASELINE`

阶段2推荐方案已于 2026-08-25 由用户确认。正式研究快照为 `D:/Hulucoding/AmAzing_Data/research_snapshots/csi300_2014_present_v2/`（研究起点 2014-01-02）。2026-08-31 已按执行账本规则重冻：成员面板仍按当日成分切片；成交与估值使用 `standardized/execution_bars.parquet` / `execution_status.parquet`；成分进出审计在 `audit/membership_spell_audit.parquet`。面板数据门禁为 `DATA_GATE_PASSED`（行业覆盖率 98.22%，未映射 WARNING 13,845 行）。Holdout 价格在 `sealed/holdout/`，读取需要一次性 `sealed/holdout.capability`，并写入 `sealed/holdout.access.log`。这不是独立 OS 用户、HSM 或加密保险库。不得把可加载张量当成正式策略结论。v1 raw 只读、未重采。

现行 Development OOF 为 `experiments/stage3_hs300_v2_execution_ledger` 与 `experiments/stage4_hs300_v2_execution_ledger`（协议 `w1ngman_stage4_feasibility_oof_only_v2`）。3 日和 5 日仍未通过预注册 GO。2026-08-28 成员切片回测归档在 `experiments/stage3_hs300_v2` / `experiments/stage4_hs300_v2`，不得当作现行证据。系统保持 `MODEL_NOT_VALIDATED`。未读 Holdout 表现。不得启动 RD-Agent，不得进入阶段5。

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

## 下一门禁

KEEP_1D 工程收口已完成：系统状态 API 报告 `MODEL_NOT_VALIDATED`，建议接口失败关闭。阶段4未通过，**不得进入阶段5**公式研究、RD-Agent、复杂评分器或具有确定性交易含义的建议。不得读取 Holdout 表现。若要继续研究，必须由用户提出新的研究假设并重新冻结阶段2，而不是修改本轮 GO 阈值。
