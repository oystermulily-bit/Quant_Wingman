# quant_w1ngman 分阶段实施门禁包

审计日期：2026-08-28（Asia/Shanghai）  
状态：`KEEP_1D_BASELINE`  
系统状态：`MODEL_NOT_VALIDATED`  
数据研究状态：`DATA_READY_FOR_DEVELOPMENT`  
统计门禁：`KEEP_1D_BASELINE`

阶段2推荐方案已于 2026-08-25 由用户确认。正式研究快照为 `D:/Hulucoding/AmAzing_Data/research_snapshots/csi300_2014_present_v2/`（研究起点 2014-01-02）。面板数据门禁为 `DATA_GATE_PASSED`（行业覆盖率 98.22%，未映射 WARNING 13,845 行）。Holdout 价格在 `sealed/holdout/`。不得把可加载张量当成正式策略结论。v1 raw 只读、未重采。

阶段3最小研究闭环已在 v2 快照上跑完 Development OOF（`experiments/stage3_hs300_v2`）。阶段4正式门禁（协议 `w1ngman_stage4_feasibility_oof_only_v2`）已完成：3 日和 5 日均未通过预注册 GO。系统保持 `MODEL_NOT_VALIDATED`。未读 Holdout 表现。不得启动 RD-Agent。

## 当前四类门禁

| 门禁 | 状态 | 结论 |
|---|---|---|
| 数据门禁 | `PASSED` | `HS300PanelDataManager.load` 在 v2 上通过。硬验收：每日 300、成员例外 0、OpenTR 前缀不变。WARNING：13,845 个 Development 成员日无申万一级（覆盖率 98.22% ≥ 95%）。Holdout 价格已物理隔离。 |
| 统计门禁 | `KEEP_1D_BASELINE` | 阶段4已在同一 Development OOF 上完成配对 Bootstrap、Holm 与预注册 GO。3 日和 5 日均未通过；净增量主要来自换手下降而非毛 Alpha。 |
| 组合门禁 | `REFERENCE_ONLY` | Top20等权长仓参考回测已实现；正式白名单组合和优化器属于阶段6。 |
| 产品门禁 | `PARTIAL` | 旧训练/回测网页可导入且测试通过，但正式建议 API、状态门禁、白名单与目标权重尚不存在。 |

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

## 下一门禁

阶段4未通过。不得进入阶段5公式研究、RD-Agent、复杂评分器或正式建议 API。不得读取 Holdout 表现。若要继续，必须由用户提出新的研究假设并重新冻结阶段2，而不是修改本轮 GO 阈值。
