# quant_w1ngman 分阶段实施门禁包

审计日期：2026-08-25（Asia/Shanghai）  
状态：`STAGE3_MINIMUM_LOOP_COMPLETED`  
系统状态：`MODEL_NOT_VALIDATED`  
数据研究状态：`DATA_READY_FOR_DEVELOPMENT`  
统计门禁：`PENDING_STAGE4_FEASIBILITY_DECISION`

阶段2推荐方案已于 2026-08-25 由用户确认。正式研究快照为 `D:/Hulucoding/AmAzing_Data/research_snapshots/csi300_2014_present_v2/`（研究起点 2014-01-02）。面板数据门禁为 `DATA_GATE_PASSED`（行业覆盖率 98.22%，未映射 WARNING 13,845 行）。Holdout 价格在 `sealed/holdout/`。不得把可加载张量当成正式策略结论。未启动阶段4统计可行性，未读 Holdout 表现。v1 raw 只读、未重采。

阶段3最小研究闭环已在 v2 快照上跑完 Development OOF（`experiments/stage3_hs300_v2`）。合成测试不能当作研究结论；正式 OOF 数字也不得在阶段4前用于宣布 3 日或 5 日更优。

## 当前四类门禁

| 门禁 | 状态 | 结论 |
|---|---|---|
| 数据门禁 | `PASSED` | `HS300PanelDataManager.load` 在 v2 上通过。硬验收：每日 300、成员例外 0、OpenTR 前缀不变。WARNING：13,845 个 Development 成员日无申万一级（覆盖率 98.22% ≥ 95%）。Holdout 价格已物理隔离。 |
| 统计门禁 | `PENDING_STAGE4_FEASIBILITY_DECISION` | 1/3/5日 Development OOF 已在 v2 上跑完；阶段4 Bootstrap/Holm 与预注册 GO 尚未运行。 |
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

## 下一门禁

下一门禁是阶段4：配对移动块 Bootstrap、Holm 校正、预注册 GO。不得在阶段4前宣布 3 日或 5 日优于 1 日。不得读取 Holdout 表现。不得启动 RD-Agent。
