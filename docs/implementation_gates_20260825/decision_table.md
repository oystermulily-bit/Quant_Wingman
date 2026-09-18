# 阶段 2 决策表

## 2026-09-15追加：阶段4/5联合研究S2

状态：`WORKFLOW_APPROVED / EXPERIMENT_FREEZE_PENDING`。用户已批准[联合研究S2计划](joint_research_s2_20260915.md)及工程修改；以下为独立版本记录，不覆盖下表D01–D28、D18-v2或D21-v1/v2/v3。

- **D28-S2，阶段前置（流程已同意）**：新S2不要求原阶段4周期/三信号/简单规则先成功；直接以受限内层联合研究、外层完整流程验证代替。只对S2生效，旧失败保留。
- **D21-S2，研究对象（流程已同意，配置待冻）**：已有审核特征的联合子集与固定拟合；低IC不自动淘汰，允许有限配对/回删。候选、组数、方向、模型、具体日期及 `signal_output` 必须显式冻结；不得偷用旧三信号等权作默认。
- **D18-S2，组合验证（合同待完整冻结）**：最终组合仍需净收益、相对A改善、统计与稳健性；此次不降低D18-v2经济要求。不再要求至少两个单因子分别优于旧集成。新的集中风险指标/阈值/对照必须先冻，不能看冠军后决定。
- **D19-S2，选择偏差（待冻）**：内层选择与外层评价分离；比较家族及选择偏差诊断覆盖完整研究流程。外层不得回流调参，Holdout仍封存。
- **D20-S2，统一预算（待冻）**：跨外层、进程、重启与最终重选统一累计。若以后允许受限公式生成，所有生成/筛选也只在内层并共享预算；旧D20额度不可额外叠加绕过总账。当前不启用RD-Agent。

最新历史结果说明：D21-v3独立3R已完成，4R为 `D21_V3_FAILED`，Holdout未读、Stage5未开放（见 `experiments/stage3r_d21_v3_reference_only_h5/run_state.json`）。下文v3的“待审查/待运行”保留为2026-09-14登记时的历史状态，不作当前进度依据。S2工程完成或预测文件生成均不等于真实实验已批准、SOTA接纳或研究GO。

## 旧冻结决策记录（保留）

最终状态：`CONFIRMED_FREEZE_20260825`（D01–D28 的 v1）。2026-09-01 追加 `CONFIRMED_FREEZE_20260901_R1`：保留 D21-v1 证据，新增 D21-v2 / D18-v2。细则见 `d21_v2_freeze.md`。下表“状态”列保留确认前的决策来源。

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
| D18 | 周期主指标（D21-v1） | OOF全相位中位数净Sharpe；GO需+0.15、年化+2pct、CI下界>0、MDD≤1日×1.2 | `CONFIRMED_FREEZE_20260825` | v1 阈值看结果前已确认；不得因 v1 失败改写。 |
| D18-v2 | D21-v2 绝对+相对门槛 | 成本×1 下候选 D 五个相位中位净 Sharpe>0 且扣费年化>0；相对实验 A（同为5日 NAIVE_TOP20）仍要求 ΔSharpe≥0.15、Δ年化≥2pct、Bootstrap CI下界>0 | `CONFIRMED_FREEZE_20260901_R1` | 防止“比旧1日少亏”形式上通过。对照不是 D21-v1 的1日。 |
| D19 | 多重检验 | 周期比较Holm家族5%；公式搜索BH-FDR q=0.05；全部候选计入 | `PROPOSED_FREEZE` | 控制选择偏差。 |
| D20 | 公式搜索预算 | 阶段5最多30轮×16候选=480；token≤64、深度≤9、算子≤12、单公式8s、单轮180s | `PROPOSED_FREEZE` | 无基于表现的提前停止；基础设施连续3轮失败则终止并记失败。 |
| D21 | 第一阶段参考组合（v1） | `D21-v1=SIMPLE_ENSEMBLE_V1+NAIVE_TOP20`；Top20等权、每股5%、缺口现金、不补位 | `CONFIRMED_FREEZE_20260825` | 只用于周期可行性。结论 `KEEP_1D_BASELINE` 保留，产物不覆盖。 |
| D21-v2 | 第二代第一阶段参考信号与组合 | 保留 D21-v1。新协议仅研究 H=5，冻结 `SLOW_RESIDUAL_ENSEMBLE_V1+BUFFERED_TOP20_5D_V1`；三个行业残差信号等权，Top20进入、排名跌出Top40才退出，五个相位全部运行，每股5%，阻塞缺口现金、不补位，不学习权重。必须跑消融 A–D。 | `CONFIRMED_FREEZE_20260901_R1` | 针对 v1 信号周期不匹配、换手、折不稳和5日集中度。`development_adaptive=true`。独立 Stage3R/4R。Holdout 继续封存。 |
| D22 | 正式组合约束 | 白名单/成员限制、长仓、单股5%、行业30%、允许全现金 | `PROPOSED_FREEZE` | 阶段6实施。 |
| D21-v3 | 长窗口收缩残差、科技参考池 | H=5；交易及排名限当日沪深300，新增中证500电子/计算机/通信仅作LOO参考；不补目标入池前残差；长窗口不变、三信号等权、总体覆盖≥85%；经济及稳健性沿用D18-v2，消融及执行沿用D21-v2。 | `CONFIRMED_FREEZE_20260914_R1` | 见d21_v3_freeze.md；严格预检89.76%，数据时点审查待完成，Stage3R/4R未运行，Stage5关闭。 |
| D23 | 优化失败回退 | 降风险→风险平价→等权→现金，全部带原因码 | `PROPOSED_FREEZE` | 禁止随机权重。 |
| D24 | API 核心合同 | 旧API不破坏；新 `/api/v2` 带schema/version/status/reason_codes | `PROPOSED_FREEZE` | 阶段9实施。 |
| D25 | 系统状态 | 只用命令规定枚举；当前 `MODEL_NOT_VALIDATED` | `PROPOSED_FREEZE` | 禁止“基本通过”等模糊结论。 |
| D26 | REINFORCE | 生成端永久禁用；旧类/字段只兼容读取 | `KEEP_COMPATIBILITY` | RD-Agent只提出公式，StackVM本地执行。 |
| D27 | 人工白名单 | 不回填历史主结论；可信度来自阶段10前向影子运行 | `PROPOSED_FREEZE` | 正式Holdout用冻结研究参考组合。 |
| D28 | 复杂化停止 | 3/5日、行业或三层规则未过Development OOF门禁即保留简单方案 | `PROPOSED_FREEZE` | 结果可为KEEP_1D_BASELINE/KEEP_SIMPLE_MODEL。 |

## 确认记录

2026-09-14 追加D21-v3（`CONFIRMED_FREEZE_20260914_R1`）：用户确认沪深300交易池不变，新增科技股仅作残差参考，保留长窗口、采用收缩残差、覆盖门槛85%。完整合同见 `d21_v3_freeze.md`。D18-v2经济及稳健性要求继续适用，旧D21/D21-v2行与失败证据不变。Stage3R/4R待新合同实现和数据时点审查；Stage5仍关闭。

用户已回复“确认推荐方案，进入阶段3”，等同接受所有 `PROPOSED_FREEZE` 和推荐的 `BLOCKED_USER_DECISION` 值。阶段3已按冻结值编码。

2026-09-01 用户确认 D21-v2 / D18-v2 冻结细则并要求执行 Stage 3R/4R。D21-v1 证据与 `KEEP_1D_BASELINE` 不得覆盖。Stage 4R 结论为 `INSUFFICIENT_EVIDENCE`（OOF 慢速残差覆盖率 77.7% < 95%），失败产物保留。阶段5仍关闭。
