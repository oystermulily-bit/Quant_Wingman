# 阶段 2 决策表

最终状态：`CONFIRMED_FREEZE_20260825`。下表“状态”列保留确认前的决策来源；用户已通过“确认推荐方案”接受全部推荐值，正式值见 `confirmation_record.json`。

状态说明：`PROPOSED_FREEZE` 表示建议值等待一次性确认；`BLOCKED_USER_DECISION` 表示存在真实冲突，确认前不能编码；`KEEP_COMPATIBILITY` 表示只保留旧行为而不纳入正式路径。

| ID | 主题 | 建议冻结值 | 状态 | 依据/影响 |
|---|---|---|---|---|
| D01 | 数据 Schema | 使用 `data_dictionary.md` 的版本化长表；所有表带 source/snapshot/schema/hash 时间元数据 | `PROPOSED_FREEZE` | 支持审计和重复构建。 |
| D02 | 数据源与起点 | AmazingData 为主，2010—2012 需第二点时源；不得静默缩短 | `BLOCKED_USER_DECISION` | 供应商手册称股票日线从2013。选择“补源”或“正式起点改为实际首日”。 |
| D03 | 成分股 | `000300.SH` 每日权重为主，区间表交叉核对；每日必须300，否则 Data Gate失败 | `PROPOSED_FREEZE` | 避免当前名单回填和生存偏差。 |
| D04 | 行业体系 | 申万一级，历史有效区间；行业统计 Leave-One-Out | `PROPOSED_FREEZE` | 不用当前行业覆盖历史。 |
| D05 | known_at | 原生优先；缺指数公告时不得提前使用，成员从 effective date 生效 | `PROPOSED_FREEZE` | 不把生效日伪装成公告日。 |
| D06 | available_at | 原生优先；日线缺原生时间时派生为交易日21:00 Asia/Shanghai | `PROPOSED_FREEZE` | 保守保证收盘数据完整；请确认21:00。 |
| D07 | 公司行动 | 公告与生效分离；OpenTR由PRECLOSE链本地构建；不使用bfill | `PROPOSED_FREEZE` | 必须通过前缀和现金流测试。 |
| D08 | 退市 | 不删除；无法退出且无处置价时退市日剩余仓位记100%损失，另做敏感性 | `PROPOSED_FREEZE` | 保守处理尾部风险；请确认。 |
| D09 | 调出沪深300 | 生效日起不新增、目标0；下一可卖开盘退出；阻塞期间保留PnL与风险 | `PROPOSED_FREEZE` | 返回 `UNIVERSE_EXIT_PENDING`。 |
| D10 | 标签 | `y_H[t]=log(OpenTR[t+H+1]/OpenTR[t+1])`，H=1/3/5；另建市场/行业超额标签 | `PROPOSED_FREEZE` | t收盘信号，t+1开盘执行。 |
| D11 | 信号/执行 | `feature_available_at` 后生成；t+1实际可交易检查不得回写t信号 | `PROPOSED_FREEZE` | 信号和执行证据分离。 |
| D12 | 参考成本 | 买0.00076、卖0.00126；成本×0/1/1.5；历史时变成本次级 | `BLOCKED_USER_DECISION` | 佣金0.00025、卖印花税0.0005、过户0.00001、滑点0.0005，请确认。 |
| D13 | 可交易/流动性 | 停牌禁成交；一字涨停禁买、一字跌停禁卖；20日成交额中位数≥2000万元方可新开 | `BLOCKED_USER_DECISION` | 流动性数字需确认。 |
| D14 | OOF | 5个 expanding validation folds，所有股票按完整日期同折 | `PROPOSED_FREEZE` | 比当前4个验证段更明确。 |
| D15 | Purge/Embargo | Purge=20交易日，embargo=5交易日；并验证事件区间不重叠 | `PROPOSED_FREEZE` | 覆盖最长5日标签+1日执行延迟。 |
| D16 | Holdout | 推荐尾部 `max(10%,480交易日)` 且至少两个日历年 | `BLOCKED_USER_DECISION` | 与“恰好10%”命令冲突；二选一或接受推荐调和。 |
| D17 | 三层职责 | 市场只控总风险/现金，行业只控预算/集中度，个股决定优先级 | `PROPOSED_FREEZE` | 避免同一风险重复计量。 |
| D18 | 周期主指标 | OOF全相位中位数净Sharpe；GO需+0.15、年化+2pct、CI下界>0、MDD≤1日×1.2 | `BLOCKED_USER_DECISION` | 阈值必须看结果前确认。 |
| D19 | 多重检验 | 周期比较Holm家族5%；公式搜索BH-FDR q=0.05；全部候选计入 | `PROPOSED_FREEZE` | 控制选择偏差。 |
| D20 | 公式搜索预算 | 阶段5最多30轮×16候选=480；token≤64、深度≤9、算子≤12、单公式8s、单轮180s | `PROPOSED_FREEZE` | 无基于表现的提前停止；基础设施连续3轮失败则终止并记失败。 |
| D21 | 第一阶段参考组合 | `SIMPLE_ENSEMBLE_V1`、Top20等权、每股5%、缺口现金、不补位 | `BLOCKED_USER_DECISION` | 只用于周期可行性，不是用户产品。 |
| D22 | 正式组合约束 | 白名单/成员限制、长仓、单股5%、行业30%、允许全现金 | `PROPOSED_FREEZE` | 阶段6实施。 |
| D23 | 优化失败回退 | 降风险→风险平价→等权→现金，全部带原因码 | `PROPOSED_FREEZE` | 禁止随机权重。 |
| D24 | API 核心合同 | 旧API不破坏；新 `/api/v2` 带schema/version/status/reason_codes | `PROPOSED_FREEZE` | 阶段9实施。 |
| D25 | 系统状态 | 只用命令规定枚举；当前 `MODEL_NOT_VALIDATED` | `PROPOSED_FREEZE` | 禁止“基本通过”等模糊结论。 |
| D26 | REINFORCE | 生成端永久禁用；旧类/字段只兼容读取 | `KEEP_COMPATIBILITY` | RD-Agent只提出公式，StackVM本地执行。 |
| D27 | 人工白名单 | 不回填历史主结论；可信度来自阶段10前向影子运行 | `PROPOSED_FREEZE` | 正式Holdout用冻结研究参考组合。 |
| D28 | 复杂化停止 | 3/5日、行业或三层规则未过Development OOF门禁即保留简单方案 | `PROPOSED_FREEZE` | 结果可为KEEP_1D_BASELINE/KEEP_SIMPLE_MODEL。 |

## 确认记录

用户已回复“确认推荐方案，进入阶段3”，等同接受所有 `PROPOSED_FREEZE` 和推荐的 `BLOCKED_USER_DECISION` 值。阶段3已按冻结值编码。
