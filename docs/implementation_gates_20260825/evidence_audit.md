# 阶段 1：证据化审计

状态：`COMPLETE_READ_ONLY`  
总体系统状态：`MODEL_NOT_VALIDATED`

## 已复现事实

- 全套测试：`281 passed, 17 skipped, 2 warnings`，耗时 56.12 秒。
- Web 模块可导入；`/api/health` 返回 `status=ok`、`generator_backend=rd_agent`；应用注册 47 条含文档/静态入口的路由。
- 2026-08-12 单票 114 轮冠军回测在隔离临时目录复现：PnL 1.54153、Sharpe 0.1388、Sortino 0.1636、1293 次交易；输出报告 SHA-256 与归档完全相同。
- 上述数字不是正式研究结果：它是单只 `000001.SZ`、全历史回测、旧连续多空仓位，且并非独立沪深300 Holdout。

## 完整信号链结论

| 环节 | 当前实现 | 结论 |
|---|---|---|
| MCP raw | 项目运行时没有 AmazingData 冻结快照加载合同 | 正式链路缺失。 |
| 标准化数据 | 单票 Parquet 或 MT5 数据管理器 | 不含点时成分、行业、公司行动和交易状态。 |
| Wingman 特征 | 65 个价格/量价/截面特征 | 单票可运行；不等于沪深300点时特征。 |
| RD-Agent | `W1ngmanEngine` 强制 `rd_agent` | 已生效；REINFORCE 不参与生成。 |
| StackVM | 62 算子，输出 `[N,T]` 和 invalid mask | 可复用，现有因果测试通过。 |
| 公式验证 | legacy reward + Validation Unit + LightGBM | 可运行，但协议仍是旧 1 日/tanh 组合语义。 |
| SOTA | SQLite/NPZ + 逐步前向接纳 | 代码存在；当前 `multi_symbol` 库 0 条因子。 |
| 回测 | 连续多空 `tanh(factor)`、对称固定成本 | 不符合 A 股长仓目标组合。 |
| 网页/API | 训练、回测、实时监控 | 没有正式建议/白名单/门禁状态合同。 |

## P0 阻塞

### P0-01：没有正式沪深300面板入口

`data_pipeline/parquet_manager.py:241-308` 只接受单票文件；`data_pipeline/data_manager.py:19-97` 是 MT5 多品种管理器。仓库没有 `HS300PanelDataManager`，无法表达每日成员、行业、停牌、涨跌停、退市及统一日期折。

影响：数据门禁、统计门禁、组合门禁全部不能运行。

### P0-02：2010 起点与供应商覆盖冲突

既有 AmazingData 手册审计记录股票/指数日线为 2013 年至今；请求起点是 2010-01-01。旧 `hs300_daily_2013_present` 目录当前不存在，即使存在也因 `bfill` 等问题被隔离。

影响：不得静默把 2010 改成 2013，也不得用未来数据补 2010—2012。必须由用户选择补充第二点时数据源或接受实际起点。

### P0-03：Holdout 在训练入口自动运行，时机早于模型冻结

`train_file.py:127-129` 在每次训练结束后保存策略并调用 `_run_frozen_holdout`；`train_file.py:112-116` 在检查点已达到目标时也会调用。`factor_research/release_audit.py` 能阻止同一 release key 重复覆盖，但不能证明这次访问发生在阶段 7 完整模型冻结之后。

影响：现有实现不满足“完整方案冻结后一次性 Holdout”。正式面板路径必须把 Holdout 命令与训练命令物理分开并写访问账本。

### P0-04：legacy IC 与 PnL 的标签索引不一致

`data_pipeline/data_manager.py:305-331` 已定义 `target_ret[t]=log(open[t+2]/open[t+1])`。PnL 在 `model_core/backtest.py:424-429` 使用 `factor[t] * target_ret[t]`；但 `model_core/engine.py:408-431` 和 `model_core/backtest.py:176-199` 又计算 `factor[t]` 对 `target_ret[t+1]`。

影响：同一公式的 IC 门控与收益评分预测了不同日期，日志中的 legacy reward/IC 不可作为新协议指标。阶段 3 必须统一通过单一 label registry 配对，禁止二次偏移。

### P0-05：当前仓位语义不符合 A 股产品

`strategy_manager/signal.py:35-52`、`factor_research/validation_unit.py:92-119`、`model_core/holdout.py:136-149` 都把 `tanh(factor)` 作为 `[-1,1]` 连续多空仓位。

影响：不能满足白名单外为 0、A 股只做多、现金、单股/行业上限和不可交易回退。旧回测保留兼容，新组合必须另建。

### P0-06：正式产品状态合同尚不存在

`web/app.py` 当前路由只有训练、回测、实时监控、配置和 AI 分析；代码中没有 `MODEL_NOT_VALIDATED`、`STALE_DATA`、`MISSING_WHITELIST` 或建议 schema。

影响：旧网页能启动不等于允许输出正式建议。阶段 9 前不增加具有交易含义的页面。

## P1 重要问题

1. `data_pipeline/data_manager.py:235-252` 在交集不足时 union+ffill OHLCV，并把起始缺失填 0；正式面板必须保留缺失及原因。
2. `data_pipeline/parquet_manager.py:259-271` 会删除缺失/非法行情行并对重复时间取最后一条，未保留修订和异常审计。
3. `model_core/features.py:1337-1340` 的截面排名没有平均秩处理并列值。
4. `factor_research/factor_artifact.py:94-109` 的数据指纹只哈希 OHLCVT，不包含 symbol 顺序、成员、行业、公司行动、交易掩码和 schema。
5. `factor_research/config.py:36` 只做 200 次 Bootstrap；拟议第一期协议要求至少 2000 次。
6. 当前公式筛选没有对累计候选执行 FDR/等价多重检验控制。
7. `model_core/holdout.py:40-58` 按 bar 比例机械切最后 10%，未按完整交易日、最小时长或市场状态定义。
8. `model_core/backtest.py:131-140` 只有单一对称成本；没有买卖方向印花税、过户费版本、停牌/涨跌停成交限制。
9. `factor_research/release_audit.py:151-180` 的 release 模型哈希只覆盖旧生成后端、研究协议、0.0003 成本、Holdout 比例和等权 SOTA，未包含拟议组合约束/回退/白名单合同。
10. `times.py:127` 仍含 `ffill().bfill()`；虽然没有被主模块导入，但必须标记为旧实验隔离入口。

## P2 工程与表达问题

1. `model_core/config.py` 仍保留大量 REINFORCE/熵/重启注释和已不工作的配置，容易让日志和维护者误判；生成器本身已禁用，不构成当前生成干扰。
2. `model_core/backtest.py:419` 仍写“用于 REINFORCE 梯度更新”，实际 RD 路径只把它当 legacy 评分。
3. 根 README 写 `start_web.py`，实际文件是 `run_web.py`。
4. `web/app.py:61-66` 的 CORS 为全开放；本地研究工具可运行，但正式产品需冻结安全边界。
5. `/api/config` 和 `/api/settings` 可返回本地 AI Key 字段；新 v2 合同不得回传秘密值。
6. 现有 `strategies/holdout_000001.SZ.json` 是 15% 协议且公式与 114 轮冠军不同，不能和当前 10% release 语义合并。

## 已通过但不能外推的修复

- StackVM 前缀不变性已有单元测试并通过。
- Walk-Forward builder 会保留请求 gap，现有 gap 测试通过。
- RankIC Validation helper 已处理并列值。
- LightGBM baseline/challenge 使用共同有限样本，`feature_fraction=bagging_fraction=1`。
- 同轮 SOTA 接纳已逐个更新矩阵再重评剩余候选。
- Knowledge Forest 现在记录任务、失败和下一轮父子假设。

这些只能说明相应代码修复存在，不能替代新的点时面板与正式门禁。

## 阶段 1 结论

- 旧系统工程回归：通过。
- 旧 1 日数字复现：通过，状态仅 `LEGACY_EXPLORATORY_BASELINE`。
- 数据门禁：失败。
- 正式统计/组合门禁：未运行。
- 正式产品建议资格：没有，必须保持 `MODEL_NOT_VALIDATED`。
