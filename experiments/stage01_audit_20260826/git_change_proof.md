# Git改动证明

## 审计开始时已有状态

```text
 M data_pipeline/hs300/__init__.py
 M data_pipeline/hs300_panel.py
 M docs/implementation_gates_20260825/README.md
 M docs/implementation_gates_20260825/stage3_implementation.md
 M docs/implementation_gates_20260825/stage3_test_report.json
 M experiments/phase2_hs300_live_audit_20260825/CONFIRMATION.md
 M experiments/phase2_hs300_live_audit_20260825/DESIGN.md
 M research_stage3/backtest.py
 M research_stage3/labels.py
 M research_stage3/runner.py
 M research_stage3/signals.py
 M tests/unit/test_stage3_research.py
?? .pytest-tmp-stage3-execution/
?? .pytest-tmp-stage3/
?? experiments/stage3_hs300_v2/
?? experiments/stage3_rerun_20260826_164500/
```

## 本次审计新增

```text
?? experiments/stage01_audit_20260826/
```

## 审计过程中并发出现、但不属于本任务的文件

```text
?? research_stage4/
?? run_stage4_feasibility.py
?? tests/unit/test_stage4_feasibility.py
```

上述文件的创建时间为2026-08-26 11:31—11:34；它们在本次审计开始状态中不存在，本任务没有创建、编辑或删除它们。

受跟踪文件的修改清单在审计前后相同。本次没有改动任何核心算法、配置、API、网页、测试或既有文档；只新增阶段0—1报告、复现命令和重新生成的实验产物。

测试运行使用pytest临时目录；这些临时目录在审计前已存在，未删除以避免破坏用户资产。

## 2026-08-28授权的后续口径变更

用户随后明确要求把正式研究起点改为2014。本次后续变更涉及：

```text
data_pipeline/hs300/config.py
data_pipeline/hs300/finalize.py
docs/02_第一期实验协议.md
docs/implementation_gates_20260825/frozen_design.md
experiments/stage01_audit_20260826/（口径更新说明和复现命令）
tests/unit/test_hs300_research_start.py
```

这些变更只统一研究区间和门禁，不重写快照、标签、Holdout或既有基线收益。
