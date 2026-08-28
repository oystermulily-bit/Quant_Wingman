# 阶段3最小研究闭环实现

状态：`STAGE3_MINIMUM_LOOP_COMPLETED`  
系统状态：`MODEL_NOT_VALIDATED`  
统计门禁：`PENDING_STAGE4_FEASIBILITY_DECISION`

## 已实现

- `data_pipeline/hs300_panel.py`：冻结快照、SHA-256、原始请求证据、每日成员数和权重、OHLCV、方向交易状态、历史行业区间及 2014-01-02 覆盖门禁。
- `research_stage3/protocol.py`：尾部 `max(10%,480交易日)` 且至少两年 Holdout；Development 内 5 折 expanding OOF、Purge 20、Embargo 5。v2 切分读 `standardized/trading_calendar.parquet`（含密封 Holdout 日期、不含价格），并与 `splits/holdout.lock.json` 对账。
- `research_stage3/labels.py`：严格同索引的 1/3/5 日 `OpenTR[t+1] → OpenTR[t+H+1]` 绝对、市场超额和行业超额标签。行业 Leave-One-Out 在单成员行业上写 NaN，不除零。
- `research_stage3/signals.py`：因果 `MOM_20`、`REV_5`、`LOW_VOL_20`，截面平均秩、等权集成以及市场、行业 Leave-One-Out 和个股残差特征。单成员行业同样写 NaN。
- `research_stage3/backtest.py`：Top20 等权长仓、现金、方向成本、买卖阻塞、不补第 21 名及成本 0/1/1.5 倍敏感性。
- `research_stage3/runner.py`：只写 Development 特征、标签和 OOF 参考回测；不调用 RD-Agent，不读取 Holdout，不生成产品建议。
- `run_stage3_research.py`：命令行入口。

## 正式输入

冻结快照：`D:/Hulucoding/AmAzing_Data/research_snapshots/csi300_2014_present_v2/`

- `manifest_version` = `w1ngman_raw_snapshot_v2`
- `research_start` = `2014-01-02`
- 正式 Data Gate：`DATA_GATE_PASSED`（行业覆盖率 98.22%）
- Holdout 价格在 `sealed/holdout/`，标准化表不含 Holdout 日
- 旧 `hs300_daily_2013_present` 与隔离旧 parquet 不得作为正式输入

## 运行命令

```powershell
cd C:\Users\Administrator\Documents\Codex\2026-07-31\yue-2\quant_w1ngman
D:\anaconda\python.exe run_stage3_research.py `
  --snapshot "D:\Hulucoding\AmAzing_Data\research_snapshots\csi300_2014_present_v2" `
  --output "experiments\stage3_hs300_v2" `
  --required-start 2014-01-02
```

快照不完整时程序以退出码 2 结束，并只写 `DATA_GATE_FAILED` 报告。通过时输出：

```text
data_gate_report.json
split_plan.json
development_features.parquet
development_labels.parquet
development_oof_portfolio_daily.parquet
stage3_report.json
```

输出目录不会产生 Holdout Parquet。阶段4完成配对移动块 Bootstrap、Holm 校正和预注册 GO 判断前，统计状态保持 `PENDING_STAGE4_FEASIBILITY_DECISION`。

## 正式 Development OOF（已跑）

- 协议：`w1ngman_stage3_minimum_loop_v1` / `simple_ensemble_v1`
- Development：2,591 日，2014-01-02 → 2024-08-23；特征 777,300 行；标签 6,995,700 行
- Holdout：2024-08-26 起 483 日，hash 与 `holdout.lock.json` 一致；输出文件 Holdout 行数 = 0
- 系统状态不变：`MODEL_NOT_VALIDATED`

合成单测（`tests/unit/test_stage3_research.py`，11 passed）只验证工程规则，不能单独作为研究结论。
