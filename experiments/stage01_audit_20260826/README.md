# quant_w1ngman 阶段0—1审计

审计日期：2026-08-26  
项目：`C:/Users/Administrator/Documents/Codex/2026-07-31/yue-2/quant_w1ngman`  
指定数据：`D:/Hulucoding/AmAzing_Data/research_snapshots/csi300_2014_present_v2`

## 结论

源码、Python环境和指定快照均存在，足以完成资产盘点和一条**探索性**1日横截面基线复现。审计当时按旧的“2010年至今”口径发现覆盖不足；用户已于2026-08-28将正式研究起点改为`2014-01-02`，因此该覆盖差异不再是当前阻塞。指定沪深300数据链尚未接入 RD-Agent、StackVM、Validation Unit、LightGBM、SOTA、API和网页，完整产品链仍需单独设计和验证。

本次没有修改核心算法。新增内容只有本目录内的审计报告和基线复现产物。

## 文件

- `stage0_asset_inventory.md`：项目、环境和资产盘点。
- `stage1_evidence_audit.md`：完整链路追踪、问题和兼容风险。
- `baseline_summary.json`：1日基线的机器可读摘要。
- `reproduce_stage1.ps1`：复现命令。
- `git_change_proof.md`：工作树基线及本次未修改核心文件的证明。
- `baseline_reproduction/`：本次重新生成的Stage-3报告与Parquet产物。

## 验证摘要

- 2026-08-28研究起点变更后全量测试：`319 passed, 17 skipped, 2 warnings`。
- Web最小启动：根页面、`/api/health`、`/api/config` 均为HTTP 200。
- 2014正式研究门禁：通过；2010检查仅保留为历史覆盖证据。
- 2014可用覆盖口径：可运行，13,845个成员日缺少行业映射。
- 1日OOF基线：数值复现成功，但扣费后累计收益约 `-52.42%`，Sharpe约 `-0.655`。
- Holdout：未读取、未生成标签、未参与本次结果。
