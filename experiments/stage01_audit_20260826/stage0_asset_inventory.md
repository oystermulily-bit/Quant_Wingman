# 阶段0：范围与资产盘点

> 2026-08-28口径更新：用户已将正式研究起点冻结为`2014-01-02`。
> 本报告中2010覆盖不足仍作为原始数据来源的历史审计证据保留，但不再是当前
> Development门禁或新模型开发的阻塞项。

## 1. 项目地图

| 区域 | 作用 | 当前状态 |
|---|---|---|
| `model_core/` | 公式词表、65特征、StackVM、RD训练调度、验证回测、Holdout | 存在 |
| `factor_research/` | Validation Unit、固定LightGBM、SOTA、Knowledge Forest、发布审计 | 存在 |
| `data_pipeline/` | MT5/单文件Parquet/沪深300面板数据 | 存在 |
| `research_stage3/` | 沪深300的1/3/5日简单研究基线 | 存在，独立于公式链 |
| `strategy_manager/` | 策略装载、公式信号合成、交易目标 | 存在，旧单品种/多品种路线 |
| `web/` | FastAPI与静态网页 | 存在，旧训练/回测/实时监控合同 |
| `tests/` | 单元、属性和集成测试 | 184个文件；按函数扫描267项，pytest实际执行328项 |
| `checkpoints/` | 训练检查点 | 425个，约1.37 GB；含204个RD命名和221个旧命名 |
| `strategies/` | 最优公式和Holdout摘要 | 3个JSON |
| `factor_research_store/` | SOTA元数据、知识森林 | SQLite存在但全部研究表为空 |
| `data/` | 项目内单品种历史数据 | 45个文件，约3.32 MB |
| `experiments/` | 历史审计和实验产物 | 存在；工作树中已有未跟踪实验目录 |
| `backtest_output/`、`logs/` | 回测和日志 | 存在 |

项目根目录含72个以上脚本，历史下载、训练、回测和诊断入口较多，存在“正式入口”和“研究脚本”混用风险。

## 2. 核心入口

- 沪深300当前最小研究：`run_stage3_research.py` → `research_stage3.runner.Stage3ResearchRunner`。
- 单文件训练：`train_file.py` → `ParquetDataManager` → `W1ngmanEngine`。
- 旧多品种训练：`main.py`、`train_single.py`、`train_index.py`等。
- Web：`run_web.py` → `web.app:app`。
- 旧回测：`run_backtest.py`、`verify_all_strategies.py`及多个历史脚本。
- 旧策略执行：`strategy_manager/runner.py`。

## 3. Git与代码版本

- 分支：`main`。
- HEAD：`4fc39d392a50613b4e7f912d90b087a72d3eec42`。
- 工作树在审计开始前已非干净状态：12个受跟踪文件被修改，另有Stage-3实验和pytest临时目录未跟踪。
- 审计进行到2026-08-26 11:31—11:34时，工作区又并发出现未跟踪的`research_stage4/`、`run_stage4_feasibility.py`和`tests/unit/test_stage4_feasibility.py`。它们不是本次阶段0—1任务创建，未纳入1日基线实现审计，也未被删除或改写。
- 本次基线代表“上述HEAD + 当时未提交改动”，不能只用commit号复现。
- 本次用到的关键文件SHA-256：
  - `research_stage3/backtest.py`: `D87DA7F2BA6EC5051D6AC14F3712878707670833ADA91336FB639375C98F57AB`
  - `research_stage3/labels.py`: `517519DF20F026845CACC55DD5963247FF96852F3A5A4EC439EDE1632B82E04F`
  - `research_stage3/runner.py`: `ADB1224B6BC78FF874B390B9441FE1EA458C02901D0517D07F64DBD43E2B1675`
  - `research_stage3/signals.py`: `9F4A4ED315B760D186D2450EB9824D57D6F0731DBE8BED45FFB10D995FB76D88`
  - `research_stage3/protocol.py`: `4ED4DA783B599306EFF30B7A9D5F853B65DC88F6A425345E3400A29367EE535A`
  - `data_pipeline/hs300_panel.py`: `6C18AF860B0C7FAA5095C809758B44D48D75F99FF3E8B06936F2BE65DD775747`

## 4. Python环境与依赖

- 解释器：项目 `.venv/Scripts/python.exe`。
- Python：3.13.9，Windows 11，Anaconda基础运行时。
- 关键实装版本：NumPy 2.5.1、pandas 3.0.5、PyArrow 25.0.0、PyTorch 2.13.0+cpu、SciPy 1.18.0、LightGBM 4.7.0、FastAPI 0.141.1、Uvicorn 0.52.0、Pydantic 2.13.4、pytest 9.1.1、Hypothesis 6.164.0。
- `requirements.txt`存在，SHA-256为`B9185E30343485737A4E3D13CA66B814720D624D2B545101C93F15B5771BEEE5`。
- `requirements-dev.txt`不存在；测试依赖混在正式requirements中。
- requirements还包含MetaTrader5、Tushare、PyTDX、tvdatafeed Git依赖、Playwright等；它们不是当前Stage-3基线的必要运行依赖。
- FastAPI的`TestClient`在当前Starlette版本要求额外`httpx2`，requirements没有声明；实际Uvicorn服务可正常启动。

## 5. 指定数据资产

快照存在且标记为冻结：`csi300_2014_present_v2`，Schema=`hs300_pit_v2`，AmazingData SDK 1.1.9。

### 覆盖

- 请求起点：2010-01-01。
- 实际K线供应起点：2013-01-04。
- 可研究OpenTR/状态起点：2014-01-02。
- 全日历：2014-01-02—2026-08-24，共3,074个交易日。
- Development：2014-01-02—2024-08-23，共2,591日。
- Holdout：2024-08-26起，共483日，价格文件位于`sealed/holdout/`。
- 每日成员数严格300；Development成员行777,300；历史成分并集672只。
- 31个申万一级行业；行业映射覆盖率98.2188%，13,845个成员日未映射。
- 65特征张量形状`[672,65,2591]`。

### 标准化表

- `universe_membership`: 777,300行；无`date+code`重复。
- `daily_bars`: 777,300行；759,681行有行情，17,619行供应商不可用；OHLC关系、负成交量/金额检查无异常。
- `trading_status`: 777,300行；19,095个买入阻塞、18,600个卖出阻塞，已区分停牌和开盘涨跌停方向。
- `industry_membership`: 993个历史区间、700只股票、31个行业。
- `corporate_actions`: 48,127行；12,503行缺`known_at`。
- `security_master`: 800只证券，其中38只有退市日期。
- `raw_request_manifest`: 919个请求，全部标记成功。
- `development_labels`: 6,995,700行。

### 来源完整性限制

- `raw/`不是快照内实体目录，而是指向`csi300_2010_present_v1/raw`的NTFS Junction；移动v2目录不能得到自包含原始证据包。
- 777,300个成员日的`known_at`全部为空，来源明确标记为`MISSING_NATIVE_INDEX_ANNOUNCEMENT`。每日指数权重能证明当日集合，但不能证明调整公告何时被市场获知。
- 行业`known_at`使用`DERIVED_POLICY`，不是原生公告时间；只能支持约定口径，不能宣称完全点时可审计。
- 公司行动时态仍不完整；没有形成可执行的退市处置价格/冻结损失规则。
- 截至本次审计日2026-08-26，快照最后日期为2026-08-24；在没有最新完整交易日日历证明时，产品信号应按过期处理。

## 6. 其他当前资产

- 本地策略：`best_000001.SZ.json`、`best_601899.SH.json`和`holdout_000001.SZ.json`。旧策略缺少当前`generator_backend`、研究协议、数据指纹等关键兼容字段。
- SOTA SQLite：`factor_research_store/multi_symbol/research.sqlite3`存在，但`factors/formulas/experiments/validation_results/failures/tasks/hypotheses`等表均为0行；没有可用SOTA因子矩阵。
- 网页静态资源：`web/static/index.html`、`app.js`、`style.css`、`bg.js`均存在。
- `web_settings.json`存在，可能含敏感配置；审计未输出其值。

## 7. 当前功能清单

已存在：RD-Agent公式提案、StackVM公式执行、旧公式Validation/LightGBM/SOTA、失败记录、单品种策略保存、旧Holdout发布审计、旧回测、训练控制、实时监控、AI分析、Web静态页面；另有独立的沪深300简单三因子1/3/5日研究基线。

尚不存在统一实现：沪深300面板 → RD-Agent公式研究 → SOTA集成 → 个股组合建议 → 新API/网页。

## 8. 必须兼容内容

- 公式token、词表版本和StackVM语义。
- 现有检查点、训练历史和策略JSON的只读兼容；新字段应显式迁移，不能默默误用旧分数。
- SQLite研究表和因子身份协议。
- 现有`/api/*`路由及Web训练/回测字段。
- 单品种Parquet文件命名和导入导出。
- Holdout日期哈希与已冻结数据角色。

## 9. 允许新增或替换内容（后续阶段）

- 新建面板公式研究适配层，而不是让网页继续调用单文件入口。
- 增加版本化API合同、状态枚举、白名单和组合输出。
- 将固定成本、标签、时间切分和数据新鲜度统一为一个冻结协议。
- 增加原生known_at来源、退市处置和Holdout访问登记。
- 补齐全量张量级未来哨兵/前缀测试。

## 10. 缺失资产与停止条件

1. 2010—2013不具备完整可研究OHLCV/OpenTR/状态数据；按2026-08-28确认的协议，该区间明确排除在Development之外，不再要求补齐当前版本。
2. 缺少指数调整公告原生`known_at`。
3. 缺少完全可审计的行业原生`known_at`和1.78%的历史行业映射。
4. 缺少退市持仓的确定性处置规则。
5. 缺少指定快照对应的SOTA矩阵及完整公式链实验。
6. 缺少实际组合集中度明细和生产级当前建议合同。

因此阶段0更新结论是：2014研究起点的数据门禁已通过，但仍不得声称数据从2010年起可用于研究。进入完整公式模型开发前，仍需解决面板接入、成员原生`known_at`和产品链口径问题。

## 11. 审计范围

纳入：审计开始时的源码、Git、环境、配置结构、指定快照、项目内数据/策略/检查点/SQLite、Stage-3最小闭环、旧公式研究链、Web/API、测试与产物。  
排除：未读取Holdout价格、未调用LLM、未连接AmazingData/MT5/券商、未执行交易、未修改核心算法、未使用API Key。
并发出现的Stage-4未跟踪源码只记录为工作区变动，不将其研究结论并入本报告。
