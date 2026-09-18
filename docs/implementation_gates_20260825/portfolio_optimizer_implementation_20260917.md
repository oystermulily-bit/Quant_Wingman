# 白名单配置优化：实现与验收

首次实现：2026-09-17；2026-09-18更新。当前规则版本 `PORTFOLIO_RULES_20260918_V3`；方案版本 `offline_portfolio_plan_v2`（新增字段向后兼容）。[全5%与区间修复说明](portfolio_allocation_fix_20260918.md)。

## 当前可以使用的流程

打开 [本地配置页面](http://127.0.0.1:8766/portfolio)。启动命令仍是 `D:/anaconda/python.exe run_portfolio.py --port 8766`。

1. 默认选择 `offline_demo_optimizer_300_20260917`，完整显示300只合成股票、2013-06-17历史评分。
2. 自主勾选并保存白名单；读取时恢复原名单版本与评分快照。
3. 填写“每只最低目标（%）”。用户未指定统一默认值，所以此项必填。例如测试用0.1表示每只至少0.1%，不是0.1的账户比例。
4. 选择持仓未知、空仓或填写旧仓；按需填写净值、费用与执行假设。
5. 生成方案：查看正目标、区间、现金、收益估计与误差、行业、波动、费用、调仓差额及股数预览。
6. 保存采纳／未采纳／执行观察／账户净值／备注，追加记录绑定原方案和名单版本；不修改原方案。

**所选股票的成功推荐目标严格大于0；区间下限允许为0。** 缺分、预算不足、行业或波动约束冲突会返回明确错误，不删除所选股票、不静默退回等权。受阻、股数取整不足或经济上不值得交易时，实际持仓可以为0，并显示“完整建仓未完成”或“不交易，选择尚未执行”。

旧的两份评分包仍可查看和恢复名单。它们没有足够的校准历史与风险证据，因此不能直接生成新版优化方案。

## 算法与实现位置

- `strategy_manager/portfolio_models.py`：全池分数百分位到H周期毛超额收益的日期等权单调线性校准；最近504成熟信号日，至少252日；20日块、2000次抽样估计均值标准误。OOF声明与时间检查不代替上游训练过程审计。
- 同模块：最近最多504交易日内非重叠H周期收益，至少60个完整周期，Ledoit–Wolf协方差；禁止未来、过期、乱序、错误维度与无穷值。缺失风险列不补零，选中或持有该股票时拒绝收益驱动配置。
- `strategy_manager/portfolio_optimization.py`：CLARABEL凸优化，效用为收益减1倍标准误、3倍方差惩罚、本次比例交易成本；保留现金；单股5%、行业30%、年化波动12%、239交易日。H必须与评分模型证据一致。当前为固定用户预算模式，未声称接入有效的自动择时预算。
- `strategy_manager/reviewed_allocation.py`：小额调仓冻结、与保持原仓的效用比较、股数取整、最低佣金重算、可卖数量、旧仓保留、20日成交额门槛与1%容量、取整后资金风险复核；目标和预计执行结果分别输出。
- 权重区间：`strategy_manager/portfolio_explanations.py` 固定其他股票推荐目标，由现金吸收一只股票的变化；在原风险、行业、预算、最低目标、费用及资金约束内，求本期效用相对推荐方案下降不超过指定容忍度的连续区间。默认容忍度1基点，可调整，尚未通过真实回测验证。逐股区间不能任意混用，不是置信区间。取代旧版0／1／2倍标准误情景包络；推荐目标仍按1倍标准误扣减求解。
- 约束解释：输出单股上限、最低目标、预算、波动、行业及小额调仓的实际限制。最低目标等于5%时明确提示锁定；全部触顶且满足条件时提供风险厌恶系数脱离上限的局部估计。页面重新生成后比较上一次相同名单的参数、目标、区间和效用变化。
- 费用模式二选一：买入0.076%／卖出0.126%的含滑点总费率；或者明确的佣金、税费、过户费、滑点、最低佣金。连续求解使用比例近似与费用缓冲，成交模拟重算；不宣称离散订单全局最优。
- 压力情景：1.5倍费用、最大行业同步下跌10%、所有买入受阻、所有卖出受阻。受阻情景计算预计持仓、现金和风险；缺执行资料时保留未知。新增仓位往返成本为比例费率示例，不包含未知的未来最低佣金。

排名分数属于一只股票；配置目标效用属于一套权重，二者分别保留。高分、低分均可成为正目标；权重还取决于协方差、成本和用户约束。

## 配置证据接口

接口前缀 `/api/v2/offline-portfolio`。原有评分、白名单和方案读取路径保留。

- `POST /analysis/import`：`{snapshot_id, manifest_path}`。固定读取同目录 `research_inputs.json`，验证SHA-256及评分绑定。禁止Holdout、sealed与工作区外路径，不加载可执行模型。
- `GET /analysis?snapshot_id=...`：可用性、校准与风险方法、证据ID、行业和执行元数据。缺证据返回 `available:false`。
- `POST /plans`：必填已保存名单ID／版本、`minimum_weight`；可传 `analysis_id`固定证据版本。支持原成本字段、新 `cost_model`、`volatility_cap`、`risk_aversion`、`no_trade_band`、`account_value`及持仓 `quantity`。新增 `range_utility_tolerance_bps`（0至100，默认1，单位为基点）仅控制区间；返回 `allocation_diagnostics`、逐股 `interval_status`、区间方法及容忍度，随方案与原请求不可变保存。
- `GET /plans/{id}`：不可变方案。所有比例使用0至1小数。`holdings:null`为未知，`[]`为已确认空仓，不能互换。
- `GET /validation?snapshot_id=...`：当前快照已登记回放报告。HTTP不接受用户自报“验证通过”的报告。
- `POST /plans/{id}/events`、`GET /plans/{id}/events`：追加与读取后续观察。字段 `kind,observed_at,account_equity?,execution_status?,note`；时间必须带时区、在方案创建之后、不晚于当前时刻且有序。净值变化不直接归因于算法，可能含出入金。

算法错误使用HTTP422和 `{code,message,details}`；例如 `MINIMUM_WEIGHT_REQUIRED`、`SELECTED_SCORE_MISSING`、`MINIMUM_BUDGET_INFEASIBLE`、`RISK_HISTORY_MISSING`。失败不保存一份伪成功方案。

持久化仍使用 `portfolio_demo_store/portfolio.sqlite3`，新增不可变配置证据、回放报告及追加观察表。已存在评分、名单和旧方案保留。

## 示例证据

`scripts/build_optimizer_demo.py` 独立生成300只合成证券，用实际最小二乘拟合一个冻结公式，先拟合、再用504个成熟样本外信号日校准，并生成100个非重叠5日风险周期。

- 评分：`experiments/offline_demo_optimizer_300_20260917/manifest.json`
- 证据：`experiments/offline_demo_optimizer_inputs_20260917/manifest.json`
- 模型为合成数据上的线性公式，与原LightGBM示例分开；不是替换正式研究模型。未读取真实行情或Holdout。

## 回放复核与尚缺的真实证据

`strategy_manager/portfolio_validation.py` 和 `scripts/run_portfolio_validation.py` 提供按已登记快照、校准器和名单的顺序回放。

三个方法使用相同名单、风险硬约束和执行账本：优化器、受约束等权、受约束逆波动。成本×0／×1／×1.5分别运行。账本使用实际模拟权重和现金，受阻旧仓与退市损失不消失，禁止省略持仓收益的时间间隙。预测与风险在信号时点冻结，账户权重在下一执行开盘计价，随后收益仅用于结算。

比较模块检查全部H相位、五个外层折、同日同名单配对、净Sharpe增量、20日块2000次配对区间、收益、回撤、折／相位一致性及成本压力。跨相位抽样共用日期块，不把重叠相位当新增独立样本。缺少排除最强年份／最大行业的独立重跑或风险暴露匹配证据时，保持“证据不足”。CLI目前不会自动生成这些额外研究证据。

本轮已执行工程示例：

```powershell
D:/anaconda/python.exe -B scripts/run_portfolio_validation.py --synthetic-demo --output experiments/offline_demo_optimizer_validation_20260917_r1
```

共9组回放，每组5个交易日，只有一个合成折和一个相位。因此报告为 `INSUFFICIENT_EVIDENCE`，不能据此声称算法优于等权。最终报告位于 `experiments/offline_demo_optimizer_validation_20260917_r1/validation_report.json`，页面“配置算法回放复核”仅展示绑定当前配置证据版本的报告。重复执行请使用新的输出目录；之前的工程试验记录保留。

历史回放首版为比例成本／权重账本，明确拒绝绝对最低佣金或账户股数参数；单份方案已具备股数与最低佣金检查，但完整逐日股数撮合回测、真实停复牌／涨跌停事件、日常强制退出调度、风险暴露匹配与集中依赖独立重跑仍需后续研究接入。人工前向记录已可保存读取，尚非自动行情驱动的连续影子账户。

现有真实研究评分包没有本规则所需的完整时间隔离证据，因此本轮未运行真实市场五折全相位验证。正式建议接口继续返回 `MODEL_NOT_VALIDATED`，生产与Holdout门禁不改变。

## 验收

使用项目本地 `.demo_runtime` 依赖，新增测试覆盖正目标与不等权、约束冲突、未知持仓、受阻旧仓、股数取整、最低佣金、不交易、校准时点泄漏、全池覆盖、证据绑定、接口持久化、前向记录与回放结算。

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) '.demo_runtime'
D:/anaconda/python.exe -B -m pytest tests/unit/test_reviewed_allocation.py tests/unit/test_offline_allocation.py tests/unit/test_portfolio_workflow.py tests/unit/test_portfolio_demo_300.py tests/unit/test_api_v2_keep_1d.py tests/smoke/test_dashboard_removed.py -q -p no:cacheprovider
```

验收结果：上述101项测试通过，包含全部300只同时选中时的正目标、行业、波动和资金约束，以及配置证据ID的快照／规则绑定。浏览器实测完成300行展示、勾选保存后恢复、正目标配置、预算不可行报错与后续备注写入读取。

旧等权函数只作为命名明确的历史参考保留，页面与POST方案接口统一调用新版优化器。
