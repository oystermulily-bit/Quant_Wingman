# 离线股票选择与配置流程（2026-09-16）

> 这是2026-09-16的历史实现记录。2026-09-17配置接口已切换至正目标的收益／风险／成本优化；下文旧等权、允许所选零目标和原请求示例不再适用于新方案。请使用[新版实现、接口与验收说明](portfolio_optimizer_implementation_20260917.md)。评分包、原名单版本和旧方案仍保留可读。

状态：`OFFLINE_WORKFLOW_IMPLEMENTED / PRODUCTION_NOT_ALLOWED`。

已实现本地评分包导入、历史日期／折选择、全量成员排序、用户勾选、版本化白名单、配置模拟、恢复与方案 JSON 下载。当前内置数据全部为合成历史回放，不是真实沪深300行情或策略收益验证。正式建议仍返回 `MODEL_NOT_VALIDATED`。

上游评分包遵循[离线演示交接合同](offline_demo_fit_contract_20260915.md)。下游只读取该合同的固定评分与非执行 JSON 证据，不改写原始评分包，不向研究端回传用户选择。本文件记录已经实现的离线行为，不替代正式研究和产品合同。

## 1. 启动与页面

在仓库根目录、已安装依赖的 Python 环境执行：

```powershell
python run_portfolio.py --port 8766
```

本机已准备的解释器也可直接运行：

```powershell
D:/anaconda/python.exe run_portfolio.py --port 8766
```

启动器优先使用仓库 `.demo_runtime` 中的可选本地依赖，绑定 `127.0.0.1`。打开 [离线股票配置](http://127.0.0.1:8766/portfolio)；独立应用的 `/` 也显示该页面。`GET /api/health` 返回 `mode: "offline_portfolio"` 和 `production_allowed: false`。此启动方式不启动旧训练或交易服务。

主训练应用也已挂载同一 `/portfolio` 页面和离线 API；其顶部有“离线股票配置”入口，沿用主应用实际端口。页面流程为：选择评分包 → 选择已有日期／折 → 查看全部成员并勾选 → 保存白名单 → 设置模拟条件 → 生成及下载方案。

页面不默认勾选全体股票。读取已保存白名单会恢复其原评分快照；修改日期、名单或模拟条件后，需要重新保存选择或重新生成方案。

## 2. 评分包与全量展示

默认只显式注册以下两个包，不扫描其他实验目录：

- `experiments/offline_demo_portfolio_300_20260916/manifest.json`：300只合成代码、40个历史日期、12,000条评分。最新 `2015-12-30 / demo_outer_2` 为300个成员、299个有效分数；`SYNTHETIC_299` 缺失分数保留在末尾。
- `experiments/offline_demo_factors_20260915_r1/manifest.json`：原16只合成小样例。页面和响应明确标记其不是300只完整成员列表。

两包都通过实际拟合产生已有日期分数；当前 `signal_output` 为 `model_prediction`，即联合模型原始预测评分。该分数不能解释为单个公式的收益率、收益概率、IC或账户权重。模型／公式指纹绑定评分来源，不能据此声称已保存可供新日期复用的模型。

导入必须显式给出工作区内 `offline_demo_*` 目录中的 `manifest.json`。禁止工作区外、Holdout或sealed路径及文件跳转。只读取固定名称 `scores.parquet`、`selected_features.json`、`formulas.json`、`model_evidence.json`；不采用manifest附带的任意路径。

导入检查包括：包类型与版本、完成状态、`synthetic/development`角色、非生产标记、四类SHA-256指纹格式、全部固定文件哈希、证据与manifest一致性、七列schema、原始网格行数、成员覆盖数与覆盖率、键唯一、时间与上海日期一致、信号严格早于 `2024-08-26T00:00:00+08:00`、有限分数与有效性标记一致。文件校验不能替代上游对训练隔离和真实输入来源的审计。

每次必须选择明确的日期和外层折。保留该日期／折的全部 `is_member=true` 行：有效分数降序，同分按代码升序；无效成员按代码排列在末尾，`score`和`rank`均为空并给出原因。非成员不进入可选名单。不同折的分数不混排。

`development`包每个日期／折必须恰有300个成员，否则拒绝导入；合成包可以用于小规模演示，并明确显示实际成员数。`is_complete_universe=true`仅表示当前快照有300个成员，不表示这些数据真实或研究通过。

## 3. 持久化与版本

SQLite 位于 `portfolio_demo_store/portfolio.sqlite3`，包含评分包元数据、不可变评分快照、白名单版本和不可变方案。它独立于原始评分包与正式研究存储。

- `bundle_id`绑定导入manifest内容；相同包重复导入不新增重复记录。
- `snapshot_id`绑定评分包、信号日期与折，保存当时成员、排序、有效性、分数与来源指纹。源包后来改变或暂时不可用时，已经保存的快照仍可恢复。
- 新白名单从版本1开始，记录 `snapshot_id`及股票代码。代码必须唯一并属于所见快照；选中无效分数行不会使它获得正目标权重。
- 更新必须同时提交 `whitelist_id`和 `expected_version`。版本冲突返回409；旧版本不覆盖。空名单可以保存，但不能生成配置方案。
- 每次生成方案保存新 `plan_id`、原请求、白名单版本和快照引用；相同输入的权重计算确定，方案ID和创建时间仍是独立记录。

## 4. 实际 API

所有以下路径以 `/api/v2/offline-portfolio` 为前缀。写请求使用 `application/json`；浏览器POST须来自本机同源页面，无 `Origin` 的程序请求允许。接口拒绝未知请求字段和非法数值；输入错误使用400／422，不存在记录为404，版本冲突为409，跨源为403。公开处理错误不回传内部文件路径。

### 包与评分

- `GET /bundles`：返回 `{bundles: [{bundle_id, label, data_role, status}], warnings, production_allowed: false}`。
- `POST /bundles/import`：请求 `{manifest_path}`；成功返回与单包查询相同的元数据。
- `GET /bundles/{bundle_id}`：返回包标识、角色、状态、指纹、行数、来源信息及 `periods: [{date, fold_id, member_count, valid_count, available_at}]`、警告。来源字段含 `signal_output`、`selected_feature_count`、`formula_count`和 `score_semantics`。
- `GET /scores?bundle_id=...&date=YYYY-MM-DD&fold_id=...`：三个参数必填。返回 `snapshot_id, bundle_id, date, fold_id, data_role, expected_member_count, member_count, valid_count, is_complete_universe, available_at, rows, warnings, status`与非生产标记；`rows`每行含 `code, score, rank, signal_valid, score_available_at, reason_codes`。
- `GET /snapshots/{snapshot_id}`：恢复保存的完整评分响应，不重新选日期或折。

### 白名单

- `POST /whitelists`：请求 `snapshot_id, name, codes`，更新时另带 `whitelist_id, expected_version`。返回完整对象，含 `whitelist_id, version, name, codes, snapshot_id, bundle_id, date, fold_id, data_role, created_at, status, production_allowed`。
- `GET /whitelists`：返回 `{whitelists: [...]}`，每条为一个名单最新版本的完整对象。
- `GET /whitelists/{whitelist_id}?version=...`：指定版本时读取该版本，省略则读取最新版本。

### 配置方案

- `POST /plans`：必填 `whitelist_id`和正整数 `whitelist_version`；可选输入见下文。返回并保存方案及 `plan_id`。
- `GET /plans/{plan_id}`：读取原方案和原输入，不重新计算。

`POST /plans`输入：

```json
{
  "whitelist_id": "使用保存接口返回的ID",
  "whitelist_version": 1,
  "market_budget": 1.0,
  "holdings": null,
  "industries": {},
  "account_value": null,
  "buy_cost": 0.00076,
  "sell_cost": 0.00126,
  "assume_tradable": false,
  "tradability": {}
}
```

- `market_budget`为0至1的股票总预算。API比例使用小数；页面百分数会转换后提交。
- `holdings=null`表示持仓未知；`[]`表示已确认空仓。已知持仓每行必填 `code,current_weight`，可选 `industry,can_buy,can_sell`。权重为0至1且合计不得超过1，代码不重复；白名单外的旧持仓也可提交。
- `industries`是代码到行业标签的映射，未填写统一为同一 `UNKNOWN`行业。
- `account_value`可省略；填写时为正数，仅用于比例对应的金额展示，不生成股数或订单。
- 成本按成交金额比例：默认买入0.076%、卖出0.126%；API允许0至0.1，前端输入单位为百分比。
- `tradability`是代码到 `{can_buy,can_sell}`的映射，字段为布尔值或空。`assume_tradable=true`只为本次模拟补充“其余可交易”假设；明确的false始终优先。

方案状态为 `OFFLINE_ALLOCATION_SIMULATION_NOT_VALIDATED`，方法 `CAPPED_EQUAL_WEIGHT`，`production_allowed=false`、`orders_submitted=false`。返回标识、日期／折、白名单版本、原请求、风险上限、现金、预计换手／成本、条件执行模拟与原因码。

主要汇总字段：`stock_weight, cash_weight, cash_after_estimated_cost, estimated_turnover, estimated_cost, estimated_buy_weight, estimated_sell_weight, projected_cash_weight, simulated_execution_cost`。

`positions`每行含 `code, score, industry, selected, current_weight, target_weight, delta_weight, action, execution_status, can_buy, can_sell, executable_delta_weight, projected_weight, target_value, delta_value, reason_codes`。所有权重和费用使用调仓前账户净值为分母。

## 5. 已实现的配置与执行语义

配置采用确定性的受约束等权：仅向用户选中的有效评分成员分配正目标；单股不超过5%，单行业不超过30%，股票合计不超过输入预算。无法使用的预算留现金，允许100%现金。分数只影响排名，不按其大小推导收益或权重；没有实现预期收益、协方差、相关性风险优化或自动市场／行业预测。

所有未知行业合并到 `UNKNOWN`，共同受30%上限约束。只选5只股票时，单股上限使目标股票合计最多25%；这不是把所选股票自动等分为100%账户资金。

`holdings=null`时，只输出目标权重；当前权重、增减差额、换手、费用与执行后持仓为空，动作是 `TARGET_ONLY`，不猜测当前空仓。已知持仓后才计算计划差额；`estimated_turnover`是各股绝对权重变化之和，买卖费用分向计算。

`cash_weight`是目标现金，包含预留费用；`cash_after_estimated_cost`才是目标调仓预计扣费后的现金。若目标股票加预计费用超过账户净值，计算器降低股票预算以预留费用，不生成负现金。

目标与可执行模拟分别保存：

- 白名单外或选中但分数无效的股票，其目标为0；原实际持仓仍保留在结果行中。
- 缺少任何需买卖股票的执行信息时，标为 `UNKNOWN`，整份方案不输出假定成交后的持仓／现金；不会把未知等同于可成交。
- 执行信息已知时，先模拟可卖部分，再在现金、单股、行业和总预算剩余空间内模拟买入；只对模拟成交部分收费。
- 明确不可卖的旧仓保持原权重，标 `BLOCKED_SELL`，继续消耗相同风险预算。未选旧仓不能因目标为0就被视为已清仓；受阻旧仓已有超限会披露，不能通过新增买入扩大该超限。
- 买入可能标 `BLOCKED_BUY`、`PARTIAL_CASH_LIMIT`或 `PARTIAL_RISK_LIMIT`。无交易变化为 `NO_CHANGE`；`READY`仅表示在给定输入下可模拟执行。
- 条件模拟满足：模拟后股票权重 + 模拟后现金 + 模拟成交费用 = 1。页面展示计划差额与模拟可执行差额，二者可能不同。

## 6. 验收与限制

本轮合并验收 **68 passed**，覆盖三个新增测试文件和旧接口／界面兼容检查：

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) '.demo_runtime')
D:/anaconda/python.exe -B -m pytest tests/unit/test_portfolio_workflow.py tests/unit/test_offline_allocation.py tests/unit/test_portfolio_demo_300.py tests/unit/test_api_v2_keep_1d.py tests/smoke/test_dashboard_removed.py -q -p no:cacheprovider
```

覆盖导入schema／哈希／状态／日期隔离、300行与无效成员保留、白名单版本冲突与快照恢复、方案持久化、持仓缺失、成本现金、未知执行、买卖阻塞、冻结旧仓风险占用，以及正式建议继续拒绝。工程验收不证明盈利、市场有效性或正式发布资格。

浏览器端已实际操作验收：完整300行与299个有效分数、勾选保存、连续版本更新、刷新后恢复原名单与快照、参数变更清除旧方案、缺分股票零目标、空仓建仓，以及卖出受阻旧仓占用行业额度。五只有效股票的目标各5%、现金75%；另选缺分股票仍为零目标。16只旧包显示真实样本数，不冒充300只。最终页面无浏览器控制台错误。

当前限制：只有合成历史回放；未接入真实沪深300评分包；不能对新日期推理；未连接实时行业／流动性／停牌／涨跌停校验、券商或订单系统；没有自动交易和连续前向影子盘。Development导入能力不表示已经运行真实数据。正式 `/api/v2/portfolio/recommendation`保留拒绝骨架，旧Top20研究、D21失败记录和Holdout门禁不因离线流程实现而改变。
