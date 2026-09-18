# 阶段1：证据化审计与1日基线复现

> 2026-08-28口径更新：正式研究起点已由用户确认改为`2014-01-02`。
> 下文2010门禁失败是审计发生时对旧协议的历史复现，不再是当前阻塞；
> 2014门禁已重新验证为`DATA_GATE_PASSED`，保留13,845行行业未映射WARNING。

## 一、首要结论：目前有两条没有接通的链

### A. 指定沪深300快照实际执行链

`csi300_2014_present_v2`
→ `HS300PanelDataManager`
→ `SimpleSignalBuilder(MOM_20, REV_5, LOW_VOL_20)`
→ `LabelRegistry(1/3/5日)`
→ `DateSplitProtocol`
→ `ReferenceBacktester(Top20等权)`
→ JSON/Parquet实验文件。

证据：

- `data_pipeline/hs300/adapter.py:20-21`明确写着适配器“Not wired into W1ngmanEngine or RD-Agent”。
- `research_stage3/runner.py:107-119`明确限定Stage-3不能调用RD-Agent或读取发布Holdout。
- `research_stage3/signals.py:28-30,124-130`构造三个简单信号并等权平均。
- `research_stage3/runner.py:13-18`只导入面板管理器、简单信号、标签、切分和参考回测。

该链在实验文件处终止，没有进入RD-Agent、StackVM、Validation Unit、LightGBM、SOTA、旧策略执行、API或网页。

### B. 旧公式产品链

单个Parquet/MT5数据
→ `FeatureRegistry`
→ 旧1日`target_ret`
→ RD-Agent
→ StackVM
→ Validation Unit
→ 固定LightGBM
→ SOTA/Knowledge Forest
→ 等权公式或`tanh`信号
→ 旧回测/策略执行
→ 旧API/网页。

证据：

- `data_pipeline/parquet_manager.py:241-284`是单品种Parquet管理器。
- `model_core/engine.py:606-665`声明并构造RD-Agent→本地VM→验证→LightGBM→SOTA闭环。
- `model_core/engine.py:825-837`强制生成后端只能是RD-Agent。
- `factor_research/research_loop.py:113-180`接收候选、执行因子身份和验证准备。
- `strategy_manager/runner.py:411-468`在旧运行时对各公式`tanh(factor)`后做算术平均。
- `web/app.py:699-732`网页训练仍调用旧训练管理接口。

因此，用户要求追踪的单条端到端链在当前项目中**并不存在**。这是进入下一阶段前的P0集成阻塞，不应靠文档描述成已经实现。

## 二、逐项审计

| 检查项 | 结论 | 证据/影响 |
|---|---|---|
| 多股票面板 | 数据层支持，模型链未接通 | 张量`[672,65,2591]`；适配器未接Engine |
| 单股票Parquet入口 | Web训练仍是 | `ParquetDataManager`和`/api/training/start` |
| 时间轴对齐 | Stage-3按完整日期统一切分 | `research_stage3/protocol.py:126-207` |
| 当前成分股回填 | 未发现固定当前300只回填；但公告时态不可证明 | 每日严格300、并集672；所有成员`known_at`为空 |
| 行业点时 | 历史区间连接存在，但只使用派生known_at | `hs300_panel.py:706-758`；覆盖98.2188% |
| 特征前缀不变 | 简单信号和VM测试通过；65特征全量证据不足 | pytest含VM/Stage-3未来哨兵测试；快照报告只抽12只 |
| 65特征缺失处理 | 存在潜在历史污染 | `features_wingman.py:97-105`先把缺失行情置0再跑滚动算子，只在输出端以当前有效掩码过滤 |
| 标签/执行延迟 | Stage-3正确；旧IC错位 | Stage-3为`t+1`开盘至`t+h+1`开盘（`labels.py:50-78`）；旧`target_ret[t]`已是`t+1→t+2`，但`engine.py:420-430`又用`factor[t]`对`target_ret[t+1]`计算IC |
| Walk-Forward边界 | Stage-3正确预留 | 5折、每折purge=20；Holdout前结束 |
| Purge/Embargo | Stage-3明确；旧研究也有固定协议 | `protocol.py:175-196`；旧LGBM为purge20/embargo5 |
| 10% Holdout | 数据物理分离，Stage-3未读 | 483日，2024-08-26起；输出不含HoldoutParquet |
| Holdout重复访问 | 治理仍不够强 | Stage-3文件可被直接读取；`seal.py`使用代码内固定确认短语。旧单品种发布审计有不可变SQLite，但未覆盖该快照链 |
| RankIC并列值 | 已修复 | `validation_unit.py:25-45`和Stage-3均用average rank |
| 交易成本 | 各链不统一 | Stage-3买0.076%、卖0.126%；旧研究/网页默认单边0.03% |
| 不可交易状态 | Stage-3有方向性状态 | 19,095买阻塞、18,600卖阻塞；回测有blocked计数 |
| 退市处置 | 未实现 | 仅保存`delist_date`，没有最后可成交价、强平、冻结损失或核销规则 |
| `tanh(factor)`仓位 | 旧链仍直接使用 | `strategy_manager/signal.py:35-52`；不属于Stage-3 Top20基线 |
| SOTA逐因子边际检验 | 代码已改为逐步前向接纳 | `research_loop.py:233-280`接纳一个后立即更新矩阵并重评 |
| LightGBM随机抽样 | 已固定 | `lgbm_baseline.py:31-50`固定种子、deterministic、单线程；配置feature/bagging fraction均1.0；共同有限样本在:115-152 |
| 搜索预算 | 明确 | 30轮×16候选=最多480个提案；token64、深度9、算子12、单公式8秒、单轮180秒 |
| 跨轮评分可比性 | 已改成固定尺度，但未用历史参考分布 | `retention_policy.py:61-83`；不再使用本轮百分位 |
| 失败公式记录 | 已实现 | `research_loop.py:123-146,233-245`和`knowledge_forest.py:84-140` |
| 公式集成 | 口径不统一 | Holdout先平均原始因子再tanh（`holdout.py:118-148`）；实盘先逐公式tanh再平均（`strategy_manager/runner.py:443-460`） |
| API语义 | 只有旧路由 | 无面板/SOTA/白名单/组合建议版本化接口；Stage-3结果不被API读取 |
| API Key安全 | 存在暴露 | `web/app.py:346-348`返回全部settings；`:390-406`在`/api/config`返回`ai_api_key`字段 |
| 数据过期判断 | 未实现正式状态 | 项目未找到`STALE_DATA`生产返回；快照止于2026-08-24 |

## 三、数据门禁结果

### 历史覆盖检查（旧2010-01-01口径）

重新调用`HS300PanelDataManager(..., required_start='2010-01-01')`：

- 状态：`DATA_GATE_FAILED`。
- 致命项：`REQUESTED_START_NOT_COVERED`，首个可研究交易日为2014-01-02。
- 警告：`INDUSTRY_UNMAPPED`，13,845行。

结论：不能把该快照描述为“2010年至今可研究数据”。自2026-08-28起，正式研究区间明确从2014-01-02开始，因此此项不再阻塞当前版本。

### 可用覆盖口径（2014-01-02）

- 数据清单、文件哈希、每日300成员、重复键、OHLC和方向交易状态通过。
- 17,619个成员日无行情，均标记`UNAVAILABLE_FROM_VENDOR`，未填充OHLCV。
- 919个原始请求均在清单中标记成功。
- 快照自带的OpenTR前缀测试只覆盖8—12只样本，不能等同于全量672只×65特征未来哨兵测试。
- 成员原生`known_at`缺失和公司行动时态不完整，不满足最高等级点时审计。

## 四、1日基线复现

### 基线定义

这不是RD-Agent公式基线，而是当前沪深300Stage-3简单参考基线：

- 股票池：每日历史沪深300成员300只，并集672只。
- Development：2014-01-02—2024-08-23。
- OOF可执行区间：2018-04-03—2024-08-22，5个不重叠验证折。
- 信号：`MOM_20`、`REV_5`、`LOW_VOL_20`横截面百分位等权。
- 组合：每交易日选Top20，每只目标5%，次日开盘执行。
- 流动性门槛：20日成交额中位数不低于2,000万元。
- 成本：买入0.00076、卖出0.00126。
- 年化交易日：239。
- 随机种子：20260812（该确定性基线本身不使用随机训练）。
- Purge/Embargo：20/5。
- Holdout：483日，未读。

### 结果

| 指标 | 结果 |
|---|---:|
| OOF日收益观测 | 1,525 |
| IC / ICIR | 0.011045 / 0.062529 |
| IC正比例 | 52.58% |
| RankIC / RankICIR | 0.028270 / 0.147401 |
| RankIC正比例 | 57.27% |
| 不扣费累计收益 | +46.999% |
| 扣费累计收益 | -52.419% |
| 扣费年化收益 | -10.988% |
| 扣费Sharpe | -0.6553 |
| 最大回撤 | 55.394% |
| 累计标准化换手 | 1,117.56 |
| 日均标准化换手 | 73.28% |
| 平均现金权重 | 0.0033% |
| 平均持仓数 | 20.17 |
| 买入阻塞 / 卖出阻塞 | 15 / 2,235 |
| 缺失估值区间 | 2,243 |

集中度说明：目标组合在完全执行时的HHI为`20×5%^2=0.05`；当前输出没有逐股实际权重或行业权重，无法审计实际组合集中度，不能用目标HHI冒充实际结果。

结果解释：毛收益为正但被日频高换手和成本完全侵蚀。该基线可复现，但统计门禁和组合门禁均未通过，不是可部署策略。

### 复现一致性

- 本次报告数值与既有两次Stage-3运行在`1e-14`容差内一致。
- 1日标签列逐值一致。
- OOF每日组合数值逐值一致。
- Parquet二进制哈希受当前扩展列/元数据影响而不同，因此以冻结代码哈希、配置哈希及关键列逐值比较为准。
- 本次`stage3_report.json` SHA-256：`97F8F56E19826535187C7C5481CA76AD1E57E1733E828D1EC54F1C95E8A4FC78`。
- 本次OOF Parquet SHA-256：`727E448E3CE3730D997DF3C8F81068A510DE24C593E723CDD0A0E664BCC545AC`。

## 五、测试和最小启动

- 命令：`.venv/Scripts/python.exe -m pytest -q --disable-warnings`
- 结果：`311 passed, 17 skipped, 2 warnings in 71.59s`。
- 2026-08-28研究起点变更后重新执行全量测试：`319 passed, 17 skipped, 2 warnings in 116.39s`。
- 测试收集时工作区已并发出现未跟踪的Stage-4测试；因此该总数用于证明“当前工作区测试通过”，不能与只含阶段0—3的历史测试数量直接比较。
- Web：Uvicorn在`127.0.0.1:8876`成功启动。
- `/api/health`：200，后端`rd_agent`，REINFORCE关闭。
- `/`：200，返回HTML。
- `/api/config`：200，但含`ai_api_key`字段，属于安全问题。
- FastAPI `TestClient`不可用，因为当前Starlette要求未声明的`httpx2`；改用真实本地HTTP完成验证。

## 六、已确认问题（优先级）

### P0：进入下一阶段前必须解决

1. 指定沪深300数据没有接入公式研究和产品链。
2. API返回原始API Key字段，需在兼容前提下改成`has_api_key`或掩码。

### P1：研究结论前必须解决

1. 成分调整原生`known_at`全部缺失；行业known_at是派生策略。
2. 旧Engine的IC相对既有1日标签多移了一期。
3. 65特征在运行旧特征算子前将缺失历史置0，可能污染停牌后的滚动特征。
4. Stage-3与旧公式链成本口径不一致。
5. Stage-3对缺失估值区间使用0收益持有；2,243次出现会影响结果，需要冻结处置协议和敏感性分析。
6. 没有退市持仓处置规则。
7. Holdout是目录分离而非强访问控制；指定面板链没有一次性release审计登记。
8. 公式集成在Holdout和实盘的“先平均后tanh/先tanh后平均”语义不一致。
9. 实际组合集中度和行业集中度没有产出。
10. 数据新鲜度没有正式`STALE_DATA`状态。

### P2：工程整理

1. 工作树非干净，必须先冻结/提交本次真正运行的源码版本。
   审计期间还出现了并发未跟踪Stage-4源码，说明当前目录不是稳定的单写者审计环境。
2. requirements混合运行、可选和测试依赖，且缺少`httpx2`。
3. 根目录历史脚本众多，入口和实验脚本需要分层标记。
4. 快照raw目录是外部Junction，发布包不可自包含复现。

## 七、尚无法确认

- 2010—2013是否能从第二数据源补齐并与AmazingData做交叉校验。
- 指数调整公告真正的公布时间。
- 12,503条公司行动缺失known_at是否会进入OpenTR或特征计算。
- 38只退市证券在持仓期的最终可执行价格。
- 65特征全量672只股票的未来哨兵测试结果。
- 指定面板接入RD-Agent后，公式搜索、SOTA和LightGBM的实际计算成本与统计结果。
- 当前Web是否能在新合同下兼容旧策略/检查点；新合同尚不存在。

## 八、兼容风险

- 旧策略JSON缺少新协议字段，若直接加载会把不可比`best_score`混在一起。
- 425个检查点混有旧/RD命名，虽然加载有协议检查，仍需迁移清单。
- 新A股成本和仅做多组合语义与旧`tanh`多空仓位不兼容。
- 新面板标签不能继续使用旧Engine的IC错位逻辑。
- 若直接替换旧API字段，当前前端会失效；应新增版本化合同并保留显式兼容层。

## 九、阶段1决策

现有1日探索性基线已复现，2014正式研究起点门禁已通过，但“沪深300完整公式产品链”仍未复现。按照冻结流程，应先完成：

1. 执行已确认的2014研究起点，禁止2010—2013进入当前版本Development；
2. 修复成员/行业/公司行动时态证据；
3. 冻结统一成本、标签、组合和退市口径；
4. 设计并实现面板到公式研究链的适配，而不是继续用单文件Web入口；
5. 修复API Key暴露。
