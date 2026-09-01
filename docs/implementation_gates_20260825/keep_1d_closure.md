# KEEP_1D 收口（阶段4之后，不是阶段5）

状态：`KEEP_1D_BASELINE`  
系统状态：`MODEL_NOT_VALIDATED`  
RD-Agent：关闭  
Holdout：未读  
产品建议：拒绝，现金 100%，仓位为空  
现行证据：`experiments/stage4_hs300_v2_execution_ledger`（2026-09-01）  
归档：`experiments/stage4_hs300_v2`（2026-08-28，成员切片回测，不作现行结论）

阶段4已经判定 3 日和 5 日不能 GO。执行账本重冻后再次判定，结论不变。按 `docs/01_研究原则.md` 第 4 条和 `docs/02_第一期实验协议.md` 第 10 节，**不得进入阶段5公式研究**。本文件记录协议允许的工程收口，不是新的研究假设。

## 不允许

- 启动 RD-Agent / 公式搜索 / LightGBM / SOTA
- 实现 `FormulaValidationScorer` 或组合优化器
- 读取 Holdout 价格或表现
- 修改预注册 GO 阈值后宣称 3 日或 5 日通过
- 输出具有确定性交易含义的目标权重

## 已实现的收口

`/api/v2` 骨架只报告研究状态，建议接口失败关闭：

| 路径 | 行为 |
|---|---|
| `GET /api/v2/system-status` | `MODEL_NOT_VALIDATED` + `KEEP_1D_BASELINE`；`rd_agent_allowed=false` |
| `GET /api/v2/data-status` | 快照 id 与 OOF 窗口；不返回本机绝对路径或密钥 |
| `GET /api/v2/research/experiments/{id}` | 只摘要 stage3/stage4 冻结结论 |
| `POST /api/v2/portfolio/recommendation` | 空 `positions`，`cash_weight=1.0` |

旧 `/api/health`、训练、回测路由保持兼容，且仍标记为 `LEGACY_SINGLE_SYMBOL_PARQUET`。网页单票训练不得被解释为沪深300正式结论。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_api_v2_keep_1d.py -q `
  --basetemp "C:\Users\Administrator\Documents\Codex\2026-07-31\yue-2\quant_w1ngman\.pytest-tmp-v2"
```

覆盖系统状态、数据状态、未知实验、空仓位建议、路径未泄漏。现行 Stage-4 报告优先读取 `experiments/stage4_hs300_v2_execution_ledger/stage4_report.json`。

## 若要进入阶段5

必须先由用户提出**新的研究假设**，改写并重新确认 `decision_table.md` / `frozen_design.md`。本轮 GO 阈值不得因本结果改动。
