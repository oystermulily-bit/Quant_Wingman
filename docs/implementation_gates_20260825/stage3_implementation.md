# 阶段3最小研究闭环实现

状态：`CODE_COMPLETE_FORMAL_DATA_PENDING`  
系统状态：`MODEL_NOT_VALIDATED`

## 已实现

- `data_pipeline/hs300_panel.py`：冻结快照、SHA-256、原始请求证据、每日成员数和权重、OHLCV、方向交易状态、历史行业区间及2010覆盖门禁。
- `research_stage3/protocol.py`：尾部 `max(10%,480交易日)` 且至少两年Holdout；Development内5折 expanding OOF、Purge 20、Embargo 5。
- `research_stage3/labels.py`：严格同索引的1/3/5日 `OpenTR[t+1] → OpenTR[t+H+1]` 绝对、市场超额和行业超额标签。
- `research_stage3/signals.py`：因果 `MOM_20`、`REV_5`、`LOW_VOL_20`，截面平均秩、等权集成以及市场、行业Leave-One-Out和个股残差特征。
- `research_stage3/backtest.py`：Top20等权长仓、现金、方向成本、买卖阻塞、不补第21名及成本0/1/1.5倍敏感性。
- `research_stage3/runner.py`：只写Development特征、标签和OOF参考回测；不调用RD-Agent，不读取Holdout，不生成产品建议。
- `run_stage3_research.py`：命令行入口。

## 正式输入目录

目录必须包含 `manifest.json`。清单版本为 `w1ngman_raw_snapshot_v1`，且至少登记：

```text
raw_request_manifest
daily_bars
universe_membership
industry_membership
trading_status
trading_calendar
security_master
corporate_actions
```

每个文件必须有相对路径、SHA-256和Parquet行数。原始请求清单还必须指向不可变原始响应及其SHA-256。旧 `hs300_daily_2013_present` 不能作为正式输入。

## 运行命令

```powershell
cd C:\Users\Administrator\Documents\Codex\2026-07-31\yue-2\quant_w1ngman
.\.venv\Scripts\python.exe run_stage3_research.py `
  --snapshot "D:\path\to\frozen_hs300_snapshot" `
  --output "experiments\stage3_hs300_v1"
```

快照不完整时程序以退出码2结束，并只写 `DATA_GATE_FAILED` 报告。通过时输出：

```text
data_gate_report.json
split_plan.json
development_features.parquet
development_labels.parquet
development_oof_portfolio_daily.parquet
stage3_report.json
```

输出目录不会产生Holdout Parquet。阶段4完成配对移动块Bootstrap、Holm校正和预注册GO判断前，统计状态保持 `PENDING_STAGE4_FEASIBILITY_DECISION`。

## 当前限制

正式2010年至今点时快照尚未生成，因此本阶段只通过合成数据验证工程规则。合成测试不能证明信号有效，不能用于选择3日或5日，也不能启用网页交易建议。
