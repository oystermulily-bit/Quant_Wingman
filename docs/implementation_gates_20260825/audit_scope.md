# 审计范围

## 本轮授权与约束

用户要求按四类门禁和阶段 0—10 改造项目；原命令同时规定阶段 2 后必须暂停。因此本轮只完成阶段 0、1、2 的盘点、证据和设计产物。

本轮允许：

- 读取源码、配置、测试、旧实验、数据库元数据和不含密钥的资产信息；
- 运行现有测试；
- 在临时目录复现旧 1 日工程基线；
- 生成本目录的审计与冻结方案文件。

本轮不允许：

- 修改训练、特征、标签、回测、组合、网页或 API 核心实现；
- 下载或构造正式沪深300数据；
- 调用 LLM 生成新公式；
- 读取新的正式 Holdout 表现；
- 运行 RD-Agent 搜索；
- 改写用户已有文档、数据、检查点、策略、数据库或设置；
- 声称 3 日、5 日或三层模型优于 1 日。

## 已检查范围

- 入口：`train_file.py`、`train_single.py`、`main.py`、`run_backtest.py`、`run_web.py`。
- 数据：`data_pipeline/`、`model_core/holdout.py`、本地 Parquet 资产元数据。
- 公式：`model_core/features.py`、`ops.py`、`vocab.py`、`vm.py`、`rd_agent_generator.py`、`engine.py`。
- 研究：`factor_research/` 全部核心模块、SQLite 表计数。
- 组合/执行：`strategy_manager/`、`execution/`、旧回测。
- 产品：`web/app.py`、任务管理器、静态资源、启动脚本和 API 路由。
- 质量：全部 pytest；前缀、VM、Walk-Forward、Holdout、Web RD 后端等现有测试。
- 外部证据：AmazingData MCP 文件/hash、既有 MCP 修复报告和数据库设计报告。

## 未检查或无法证明

- 没有在本轮登录 AmazingData 或重新进行联网查询；2010 覆盖依据开发手册和既有只读审计。
- 没有正式 raw 快照，无法执行每日300只、行业历史归属、公司行动、退市和可交易掩码的数据验收。
- 没有沪深300正式面板，无法运行 1/3/5 日 OOF、Bootstrap、多重检验和组合门禁。
- 没有正式建议 API，无法运行建议状态和白名单端到端验收。

## 证据规则

1. 源码结论引用文件与行号。
2. 运行结论记录命令、环境、输入/输出哈希。
3. 旧单票数字一律标记 `LEGACY_EXPLORATORY_BASELINE`。
4. 已查看的旧单票 Holdout 不视为新的沪深300 Holdout；它也不得转用于正式结论。
5. 未实现、未运行和未通过必须分别写明，禁止用“基本通过”。

## 本轮交付终点

该审计完成时状态为 `STAGE_2_AWAITING_CONFIRMATION`；用户已于2026-08-25确认 `decision_table.md`，当前已进入阶段3。
