# 兼容性清单

## 必须保持的现有行为

1. `run_web.py` / `start_web.bat` 能启动现有训练、回测和实时监控研究页面。
2. `/api/training/start` 必须继续要求 `rounds`，网页训练仍锁定 `generator_backend=rd_agent`。
3. 旧策略 JSON 至少兼容读取：`symbol`、`timeframe`、`formula`、`formula_decoded`、`best_score`、`data_file`、`generator_backend`。
4. 旧训练包、检查点、训练历史和回测报告只读可导入；协议不一致时必须拒绝续训而非静默混用。
5. 现有 API 路径保持可用；新增正式研究/建议合同使用版本化新路径，不改变旧字段语义。
6. 硅基流动配置名 `SILICONFLOW_API_KEY`、`SILICONFLOW_MODEL` 和 `W1NGMAN_GENERATOR_BACKEND` 保持兼容。
7. REINFORCE 不得恢复为公式生成器。

## 允许替换或新增

- 新增只读冻结快照加载器和 `HS300PanelDataManager`，不改写旧 `ParquetDataManager` 文件格式。
- 新增 1/3/5 日标签模块、统一日期 splitter 和四类独立门禁。
- 新增 A 股白名单长仓组合与确定性回退，不复用旧 `tanh(factor)` 仓位语义。
- 新增 `/api/v2/data-status`、`/api/v2/research-status`、`/api/v2/portfolio-recommendation` 等版本化合同。
- 在阶段 9 调整网页以显示门禁、状态、目标权重和现金；旧训练/回测工作区继续保留。

## 禁止修改的历史资产

- `experiments/20260812_000001_SZ_rd_step0114/` 及其 SHA-256 清单；
- `data/` 中现有单票文件；
- `checkpoints/`、`strategies/` 和 `factor_research_store/` 中既有本地产物；
- AmazingData 原始响应或将来正式冻结的 raw 快照；
- 已经查看过的单票 Holdout 结果；
- 用户现有 `web_settings.json`，且任何审计报告不得复制其中密钥。

## 废弃但暂留的兼容入口

| 入口 | 原因 | 新产品是否使用 |
|---|---|---|
| `times.py` | 独立旧实验脚本，仍含 `ffill().bfill()`；没有被项目模块导入。 | 否；正式路径必须隔离。 |
| `train_ftmo*.py` | 外汇/MT5 奖励和训练入口。 | 否。 |
| `train_index.py`、`_launch_index.bat` | 旧指数品种训练，不是点时沪深300横截面。 | 否。 |
| `run.py`、`live_trade.py` | MT5 实盘执行。 | 否。 |
| `data_pipeline/MT5DataManager` | 多品种交集或 union+ffill 语义。 | 仅兼容旧功能。 |
| `strategy_manager.compute_target_positions` | `tanh(factor)` 连续多空。 | 仅旧回测/监控。 |

## API 兼容面

现有 36 个唯一 `/api` 路径和根页面 `/` 按功能分为：健康/配置、AI分析、数据和策略文件、训练、回测、实时行情及飞书通知；加上 OpenAPI/文档路由后应用共注册 47 条路由。阶段 9 前不删除这些路径。

新增响应必须带 `schema_version`。旧策略缺少新字段时只允许映射到 `LEGACY_RESEARCH_ARTIFACT`，不得自动升级为 `PRODUCTION_ELIGIBLE`。

## 已发现的兼容文档问题

根 README 的启动命令写成不存在的 `start_web.py`；真实 Python 入口是 `run_web.py`，Windows 包装是 `start_web.bat`。本轮不改根 README，以免越过阶段 2；产品门禁实施时修正并加测试。
