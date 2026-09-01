# 验证协议（已冻结 v1）

状态：`CONFIRMED_FREEZE_20260825`

## 1. 研究问题顺序

1. 在同一数据/成本/执行口径下，3日或5日是否相对1日有稳定 Development OOF 增量？
2. 简单行业上下文是否有边际增量？
3. 简单市场风险开关是否改善扣费风险收益？
4. 只有前三项通过，才允许 RD-Agent 公式研究和复杂组合。

不得跳过前四阶段直接拿最优公式、最好年份或最好行业作结论。

## 2. 样本与切分

- 单位是完整交易日期，不是股票行。
- 同一日期全部股票进入同一 role。
- Development 内推荐 5 个 expanding Walk-Forward OOF 验证折。
- Purge=20日且逐样本断言 `train.label_end_time < validation.feature_available_at`。
- Embargo=5日；未来 fold 的训练集不得使用前一验证窗后的 embargo 日期。
- Holdout 方案 D16 已确认为尾部 `max(10%,480交易日)` 且至少两个日历年。封存方式：只读文件 + 一次性 `sealed/holdout.capability` + 追加访问日志。这不是独立 OS 用户、HSM 或加密保险库。未到阶段8不得消耗该令牌或读取 Holdout 价格。

## 3. 第一阶段预注册信号

只使用 `MOM_20`、`REV_5`、`LOW_VOL_20`，形成 `SIMPLE_ENSEMBLE_V1`。窗口、方向、截尾、缺失、等权和 Top20 规则看到结果后不得改变。

每个周期报告所有调仓相位：

- 1日：相位0；
- 3日：相位0/1/2；
- 5日：相位0/1/2/3/4。

主结论使用全相位中位数；最好相位只进附录。

## 4. 主指标和 GO 门槛

唯一主指标：点时沪深300 Top20 等权长仓+现金参考组合的扣费 Development OOF 全相位中位数净 Sharpe。

候选周期相对1日必须同时满足：

1. 净 Sharpe 改善≥0.15；
2. 扣费年化收益改善≥2个百分点；
3. 同日期配对移动块 Bootstrap 的95% CI下界>0；
4. 3日至少2/3相位、5日至少3/5相位与中位数方向一致；
5. 最差相位净 Sharpe不得比1日低0.25以上；
6. 多数 OOF 折净收益和 RankIC方向一致；
7. 最大回撤≤1日最大回撤×1.2；
8. 至少2/3简单单信号的周期差异方向与集成一致；
9. 优势不能只来自一个年份、行业或少数股票；
10. 成本×0/1/1.5结果全部披露，并拆分 gross alpha、成本节约和净改善。

阈值已由用户确认，不得根据结果改动。

## 5. 不确定性与多重检验

- Bootstrap：移动时间块，块长20交易日，2000次，seed=20260812，95% CI。
- 周期主家族：3日对1日、5日对1日，Holm校正，family alpha=0.05。
- 阶段5公式家族：所有唯一候选进入 BH-FDR，q=0.05；重复/非法/超时同样计入搜索机会和失败登记。
- 公式搜索报告累计候选数、等价公式数、通过硬门槛数、接纳数、FDR状态、Deflated Sharpe或等价诊断、PBO或等价诊断。
- 不只报告点估计；不只展示冠军。

## 6. 必报指标

### 预测/横截面

IC、ICIR、RankIC、RankICIR、方向胜率、正收益概率及校准误差、有效样本数、覆盖率、行业覆盖率、最大行业贡献、分组收益单调性。

### 组合

毛/净收益、年化、Sharpe、Sortino、Calmar、最大回撤、换手、成本、持仓数、现金比例、行业集中度、执行阻塞、调出/退市损益。

### 稳定性

每折、每年、市场状态、行业、剔除最强行业、调仓相位、成本敏感性及其置信区间。

## 7. 阶段4门禁状态

| 条件 | 状态 |
|---|---|
| 3日通过、5日未通过 | `RESEARCH_GATE_PASSED`，只允许继续3日；报告子结论 `HORIZON_FEASIBLE_3D` |
| 5日通过、3日未通过 | `RESEARCH_GATE_PASSED`，只允许继续5日；报告子结论 `HORIZON_FEASIBLE_5D` |
| 两者通过 | `RESEARCH_GATE_PASSED`；报告 `HORIZON_FEASIBLE_BOTH` |
| 两者未通过、1日仍可用 | `KEEP_1D_BASELINE` |
| 简单规则优于复杂规则 | `KEEP_SIMPLE_MODEL` |
| 数据或统计证据不足 | `INSUFFICIENT_EVIDENCE` |
| 协议破坏 | `RESEARCH_GATE_FAILED` 且系统 `MODEL_NOT_VALIDATED` |

这些是研究结论，不等于 `PRODUCTION_ELIGIBLE`。

## 8. 阶段5公式门禁

只有阶段4通过后启动。固定预算30×16=480候选，RD-Agent只接收词表、匿名结构和 Development 聚合反馈。单公式必须先通过：合法/非退化、覆盖≥95%、FDR、跨折/年份/行业稳定、相对行业残差预测、与SOTA相关性、固定LightGBM OOF边际增量、扣费组合未恶化。

每接纳一个公式立即更新SOTA矩阵，剩余候选全部重评。跨轮 ranking 使用固定尺度/全历史参考，不使用本轮百分位。

## 9. 一次性 Holdout

阶段7生成 `FROZEN_RELEASE_CANDIDATE` 后才允许：

- 验证 release/data/config/code/SOTA/成本/组合哈希完全一致；
- 原子预留访问账本（消耗一次性 capability，写入 `sealed/holdout.access.log`）；
- 运行一次并写不可变报告；
- 不现场改公式、成本、阈值、行业规则或组合约束；
- 失败即 `HOLDOUT_FAILED`，不重跑挑最好一次。

查看后该区间不能再作为后续版本的严格独立 Holdout。

## 10. 测试门禁

必须新增：全特征前缀不变、未来哨兵、行业Leave-One-Out、成员/行业历史版本、统一日期折、1/3/5标签索引、事件区间Purge、Holdout权限、成本方向、停牌/涨跌停、退出延迟、退市损失、FDR候选计数、确定性回退、状态枚举和旧API兼容测试。
