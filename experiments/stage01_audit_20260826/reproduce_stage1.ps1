$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\Administrator\Documents\Codex\2026-07-31\yue-2\quant_w1ngman"
$SnapshotRoot = "D:\Hulucoding\AmAzing_Data\research_snapshots\csi300_2014_present_v2"
$OutputRoot = Join-Path $ProjectRoot "experiments\stage01_audit_20260826\baseline_reproduction"
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

Set-Location -LiteralPath $ProjectRoot

# 1. 记录实际源码状态。该仓库当前不是干净工作树。
git -c safe.directory=C:/Users/Administrator/Documents/Codex/2026-07-31/yue-2/quant_w1ngman rev-parse HEAD
git -c safe.directory=C:/Users/Administrator/Documents/Codex/2026-07-31/yue-2/quant_w1ngman status --short

# 2. 2026-08-28冻结后的正式研究口径数据门禁。
& $Python -c "from data_pipeline.hs300_panel import HS300PanelDataManager; import json; m=HS300PanelDataManager(r'$SnapshotRoot', expected_members=300, required_start='2014-01-02').load(raise_on_gate_failure=False); print(json.dumps(m.report.to_dict(), ensure_ascii=False, indent=2))"

# 3. 可用覆盖口径的探索性1/3/5日Stage-3复现。1日摘要见baseline_summary.json。
& $Python run_stage3_research.py `
  --snapshot $SnapshotRoot `
  --output $OutputRoot `
  --expected-members 300 `
  --required-start 2014-01-02

# 4. 完整测试。
& $Python -m pytest -q --disable-warnings

Write-Host "注意：本脚本不读取Holdout，不调用RD-Agent，不产生生产交易建议。"
