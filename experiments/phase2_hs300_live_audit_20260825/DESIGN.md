# 沪深300点时研究数据库设计（阶段2，2026-08-25 再审计）

状态：`STAGE_2_COMPLETE_AWAITING_CONFIRMATION`

审计时刻：2026-08-25 10:44 Asia/Shanghai。本文件取代 `experiments/phase2_hs300_db_design_20260821/DESIGN.md` 中与覆盖范围、行业探测、END_DATE 有关的结论；Schema 与阶段 3–13 施工顺序仍与 08-21 稿同向，冲突处以本稿为准。

本任务范围：数据库、验证报告、Wingman 面板加载接口。不启动 RD-Agent，不筛选公式，不计算策略收益，不读取 Holdout 表现。未经确认不得进入大规模下载、转换、标签或训练。阶段3 MCP 行业修复也等确认后再改生产服务器。

---

## 0. 总评

原思路方向正确，且与 `docs/01_研究原则.md`、`docs/02_第一期实验协议.md` 的数据合同大部分同向。不能按原文直接开工。

2026-08-25 已用 AmazingData SDK 1.1.9 做**行数与日期**探测（不落行情值、不建快照）。结论：

1. 请求起点 `2010-01-01` 不能被本源完整覆盖。
2. 沪深300**每日权重**至少从 2010-01-04 可用，且该月每日恰好 300 只、权重和为百分比约 100%。
3. 股票/指数**日线 OHLCV 在 2010-01 与 2012-12 为 0 行**，样本最早 K 线为 2013-01-04。
4. `get_history_stock_status`（PRECLOSE / 涨跌停 / 停牌）对 `000001.SZ` 在 2013-01 为 0 行，2016-01 为 20 行。OpenTR 不能默认从 2013-01-04 起算。
5. MCP `mcp_industry_index_info` 仍调用不存在的 `get_industry_index_info`。SDK `get_industry_base_info()` 可用：511 行，一级 31 个，代码后缀全部 `.SI`。
6. 今日（2026-08-25）是交易日且探测时刻早于 15:00，因此 `END_DATE=2026-08-24`。

不得静默把 START_DATE 改成 2013，不得用旧 Parquet 补 2010–2012，不得把权重表里的 `CLOSE` 当作未复权行情。

---

## 1. 阶段1审计（含实盘）

### 1.1 MCP / SDK

| 项 | 结果 |
|---|---|
| 服务器 | `D:/Hulucoding/AmAzing_Data/xysz/xysz/WealthManager/ad_mcp/server.py` |
| SHA-256 | `3BC7F1ADA17436BBDB2CA28FCA91AE7206BA6A2DC5F20158EBB0C1310C370EAD`（与 2026-08-20 / 08-21 相同） |
| mtime | 2026-08-20 13:21:54 |
| SDK | AmazingData 1.1.9 |
| 工具数 | 58 |
| 当前 Cursor 会话 | 未注册该 MCP |
| 登录 | 账号可用；密码**必须 strip**。带尾部空白时 SDK 会 `SystemExit(0)` |

真实 SDK 方法：

| 需求 | SDK | 实盘 |
|---|---|---|
| 交易日历 | `BaseData.get_calendar` | 8711 日，1990-12-19 .. 2026-08-25 |
| 指数权重 | `InfoData.get_index_weight` | 2010-01 有数据；单位百分比 |
| 指数成分区间 | `InfoData.get_index_constituent` | 1226 行；INDATE 2005-04-08 .. 2026-06-15 |
| 未复权日线 | `MarketData.query_kline` | 2010/2012=0；2013-01-04 起有 |
| PRECLOSE 等 | `InfoData.get_history_stock_status` | 2013-01 样本 0 行；2016-01 有 |
| 基础信息 | `InfoData.get_stock_basic` | 有 LISTDATE / DELISTDATE / IS_LISTED |
| 分红 / 配股 / 股本 | `get_dividend` / `get_right_issue` / `get_equity_structure` | 分红有 DATE_DVD_ANN、DATE_EX、税前/税后现金 |
| 公告 | `get_announcement_stock_list` | 字段含 PUBLISH_TIME；2020Q1 对 000001.SZ 为 0 行 |
| 行业基础 | `get_industry_base_info` | 511 行；**没有** `get_industry_index_info` |
| 行业成分 / 权重 / 日行情 | `get_industry_constituent` / `get_industry_weight` / `get_industry_daily` | 一级样本 `801180.SI`；行业日行情 2010-01 已有行 |

指数 `WEIGHT` 手册单位为百分比。2013-01 实盘：每日 300 只，日度权重和 min/median/max = 99.9991 / 99.9998 / 100.0006。检测规则保持：先验 `[99, 101]`；仅当明显落在 `[0.99, 1.01]` 时记为小数；否则失败，不得猜测。

### 1.2 MCP 仍须修复（确认后阶段3）

| 工具 | 问题 |
|---|---|
| `mcp_industry_index_info` | 引用未定义 `end_date`；调用不存在的 `get_industry_index_info` |
| `mcp_industry_index_constituent` | `date` 被忽略；`serialize_dataframe` 可能吃掉 `dict` |
| `mcp_industry_index_weight` / `quote` | 同上；行业日行情的日期很可能在 DataFrame **索引**上，必须 `serialize_tabular(..., index_field="TRADE_DATE")` |
| ETF / KZZ / `mcp_history_code_list` | 方法名与 SDK 不符（本任务不用） |

`mcp_corporate_action_timeline` 没有把指数 `known_at` 设成 `INDATE`。但派生过宽：配股 `max(EXECUTE_DATE, ANN_DATE)`，股本 `max(ANN_DATE, CHANGE_DATE)`。正式账本必须在快照层重算。

`mcp_total_return_open` 不用 bfill。首日 `OPEN_TR[0]=1`（按开盘归一），与本稿推荐的 `CloseTR[t0]=1` 不同。正式 OpenTR 必须从冻结 raw 重算。

训练中不得调用 MCP。批量通道推荐 SDK 客户端，方法名与 MCP 对应，记录 `sdk_version` 与 `mcp_server_sha256`。

### 1.3 覆盖矩阵（硬阻塞）

`REQUESTED_START=2010-01-01`。不得静默缩短。缺失标为 `UNAVAILABLE_FROM_VENDOR`。

| 接口 | 2010-01 | 2012-12 | 2013-01 | 2016-01 | 最近完整月至 2026-08-24 |
|---|---|---|---|---|---|
| `query_kline`（000001.SZ + 000300.SH） | 0 | 0 | 40（自 2013-01-04） | 40 | 000001.SZ 16 根 |
| `get_history_stock_status`（000001.SZ） | 0 | 0 | **0** | 20 | 未单测 |
| `get_index_weight`（000300.SH） | 6000（20×300） | 6300（21×300） | 6000 | 6000 | 4800（16×300） |
| `get_industry_daily`（801180.SI） | 20 | — | 20 | — | — |

成分区间表可追溯到 2005 年，但这是生效区间，不是公告 `known_at`，也不能代替日线。

2010-01-01 至 2012-12-31：至少 K 线与股票状态为 `UNAVAILABLE_FROM_VENDOR`。权重与部分行业指数行情可能有值，**不能**据此声称 OpenTR 或标签从 2010 年起生产级完整。

### 1.4 严格研究起始日

按你列的 8 项，取**均达到生产级完整性的最晚日期**。当前只能给下界，不能冻结：

| 项 | 观察到的最早可用 | 生产级？ |
|---|---|---|
| 1. 可靠复权/因果日线 | K 线 2013-01-04；PRECLOSE 样本 2013-01 缺失 | 否，直到状态接口起点查清 |
| 2. 点时沪深300成员 | 权重 2010-01-04，每日 300 | 成员层可用；无公告 known_at |
| 3. 点时行业分类 | 一级样本 INDATE 可早至 1990；行业日行情 2010-01 有行 | 须确认 31 个一级均为申万、无中信混入 |
| 4. 退市与主表 | `DELISTDATE` / `IS_LISTED` | 无处置价，规则未冻结则否 |
| 5. 停复牌与交易状态 | 样本 2016-01 有，2013-01 无 | 起点未知 |
| 6. 公司行动 | 分红字段完整；样本查询从 20130101 起有 ANN/EX | 账本口径未冻结 |
| 7. 真实 available_at | 无原生字段 | 仅能 `DERIVED_POLICY`，须冻结 |
| 8. 成本与可成交性 | 涨跌停价来自状态接口 | 受第 5 项约束 |

因此：**数据库请求起点仍写 2010-01-01；可物化成员的下界约 2010-01-04；K 线下界 2013-01-04；OpenTR / 标签 / 可成交掩码的下界 ≥ max(K线, PRECLOSE)，样本显示可能晚至 2016。正式研究起点在 4/7 项规则冻结且状态覆盖查清之前不得声称。**

确认后阶段4的第一件探测（仍只记日期与行数）：对 `get_history_stock_status` 按年 1 个月窗口扫描至少 2 只历史成分股，找到最早非空年。不得在未知时默认 2013。

### 1.5 旧数据隔离

`D:/Hulucoding/AmAzing_Data/output/hs300_daily_2013_present/` 在 2026-08-25 **仍不存在**。构建脚本仍含 `adj_factor` 的 `ffill().bfill()`。只允许对照 2026-08-12 日志的数量级（724 只、约 3304 日、缺行情 19531）。禁止进入正式库。见 `QUARANTINE.md`。

### 1.6 Wingman 现接口

仍是 MT5 + 单票 Parquet：

- 无 `HS300PanelDataManager`
- 标签只有 raw `open` 的 1 日，不是 OpenTR，没有 3/5 日
- Holdout 为 `chronological_holdout_10_v1`
- `times.py` 对 OHLC `ffill().bfill()`
- `MT5DataManager` 交集过短时 `ffill` 再 `fillna(0.0)`

本任务不把面板接到 `W1ngmanEngine` / RD-Agent / Holdout 评价。

### 1.7 磁盘与时间

| 盘 | 空闲 |
|---|---|
| C: | 33.3 GB（不能做主库） |
| D: | 179.1 GB（够必需层；全 A 增强层需另批） |

`D:/Hulucoding/AmAzing_Data/research_snapshots/` 尚不存在。

必需层预估 2–8 小时、15–40 GB。增强层另加 12–36 小时、80–200 GB。

### 1.8 最近完整交易日

`TIMEZONE=Asia/Shanghai`。若当前是交易日且本地时间 < 15:00，END_DATE 为上一完整交易日。

本次探测：`today_is_trading_day=true`，`session_complete=false`，`latest_completed_trading_day=20260824`。下载当日再按同一规则冻结，不把 2026-08-24 写死进代码常量。

---

## 2. 与研究协议必须改写的冲突

| 原思路 | 协议 / 实盘 | 处理 |
|---|---|---|
| `HOLDOUT_RATIO=0.10` | 禁止机械最后 10%；≥2 日历年且 ≥480 日。约 3300 日的 10% ≈ 330 | **不用 10%** |
| 三层评分进库 | 产品待办 B；第一期行业不是硬门槛 | 只落行业统计与掩码，不算分 |
| `features[N,F,T]` 预置研究特征 | 第一期稍后才跑 MOM_20 等 | v1 `F=0` |
| 退出规则里的目标权重 / `UNIVERSE_EXIT_PENDING` | 本任务不算策略收益 | 只落成员、掩码、退出审计字段；持仓状态机属回测 |
| purge 覆盖 5 日+执行延迟 | 协议推荐 20 日 | 采用 20 日 |
| 每日必须恰好 300 | 2010-01 与 2013-01 样本每日 300 | 硬验收 + 例外表；不删行凑数 |

---

## 3. 确认后的推荐合同

见 `CONFIRMATION.md`。推荐冻结值：

```text
INDEX_CODE              = 000300.SH
REQUESTED_START         = 2010-01-01
MEMBERSHIP_START        = 2010-01-04     # 权重实盘下界（待全历史确认）
KLINE_START             = 2013-01-04     # 样本 K 线下界
STATUS_START            = UNKNOWN        # 000001 在 2013-01 为空、2016-01 非空
OPENTR_START            = max(KLINE_START, STATUS_START)
END_DATE                = latest_completed_trading_day
COVERAGE_GAP_BARS       = 2010-01-01 .. 2012-12-31  UNAVAILABLE_FROM_VENDOR
FREQUENCY               = D1
TIMEZONE                = Asia/Shanghai
INDUSTRY                = 申万一级 only（实盘 31 个 LEVEL_TYPE=1，后缀 .SI；须核名称）
ENHANCEMENT_LAYER       = OFF in v1
OPEN_TR                 = PRECLOSE 链；首日 CloseTR=1；税前隐含；全额配股
INDEX_KNOWN_AT          = null / MISSING_NATIVE_INDEX_ANNOUNCEMENT
AVAILABLE_AT            = event_date 21:00 Asia/Shanghai, DERIVED_POLICY
DELISTING               = delayed last-trade then 100% loss if never sellable
HOLDOUT                 = 含两个完整日历年且 >=480 日，purge 20
SNAPSHOT_ROOT           = D:/Hulucoding/AmAzing_Data/research_snapshots/csi300_2010_present_v1/
STATUS_AFTER_BUILD      = DATA_NOT_FORMALLY_VALIDATED until rules frozen AND tests pass
```

目录名保留 `2010_present` 表示**请求**起点；manifest 必须同时写 `requested_start`、`kline_start`、`status_start`、`opentr_start`。不得把目录名当成已覆盖 2010 年行情。

若用户坚持 2010 年完整 OHLCV：停工，不缩短，不补洞。

---

## 4. 采集清单与批次

唯一上游：修复后的 AmazingData SDK（与 MCP 同源）。`is_local=False`，参数组2：`begin_date` + `end_date`。密码入库前 strip。登录须捕获 SDK 的 `SystemExit`。

顺序：日历 → 指数年切权重 → 成分区间 → 历史 CON_CODE 并集 → stock_basic → 按年、每批 20–40 只 kline 与 **status** → 分红/配股/股本 → 可选压缩公告 → industry_base_info 一次 → 过滤申万一级后成分/日行情 → adj_factor 仅审计。

成功 raw 永不覆盖。重试写 `part=<id>.retry<n>.*`。`.partial` 校验后再 rename。

公告全量可能主导耗时。v1 推荐只拉与公司行动日期 ±20 交易日重叠的公告；000001.SZ 在 2020Q1 公告清单为 0 行，阶段4须对第二只股票复核接口是否可用。

---

## 5. 目录

```text
D:/Hulucoding/AmAzing_Data/research_snapshots/csi300_2010_present_v1/
├─ raw/
│  ├─ trading_calendar/
│  ├─ index_weight/
│  ├─ index_constituent/
│  ├─ daily_kline/
│  ├─ stock_status/
│  ├─ stock_basic/
│  ├─ dividends/
│  ├─ right_issues/
│  ├─ equity_changes/
│  ├─ announcements/
│  ├─ industry_base/
│  ├─ industry_constituent/
│  ├─ industry_weight/
│  └─ industry_daily/
├─ standardized/
├─ derived/
├─ labels/
├─ splits/
├─ validation/
├─ panel/
├─ manifest.json
├─ schema.json
├─ checksums.sha256
└─ STATUS.txt
```

仓库内只放代码与实验文档。每个 raw 响应写 `.parquet` + `.meta.json`（snapshot_id、source、mcp_server_sha256、sdk_version、method、parameters、requested_at、completed_at、response_sha256、row_count、success、retry_count）。

---

## 6. 标准化 Schema（v1）

所有表：`schema_version, data_version, source, fetched_at, snapshot_id`。

表集合与 08-21 稿相同：`trading_calendar, universe_membership, index_weights, daily_bars_raw, stock_status, stock_master, corporate_actions, adjusted_price_causal, tradability_directional, delisting_events, industry_base, industry_membership, industry_daily, industry_statistics, missing_quote_class, repair_log, audit_index_known_at`。

代码格式：`000001.SZ` / `600000.SH`。K 线实盘列为小写 `open/high/low/close/volume/amount/kline_time`，状态与权重为大写。标准化统一为大写 + `date, code`。

禁止 ffill：OHLCV、amount、PRECLOSE、涨跌停价、停牌虚构行情、adj_factor、成员资格、OpenTR。允许的状态型 ffill 必须写 Schema 理由，且同股、向未来、上市前保持缺失、有最大长度。

权重表中的 `CLOSE` 只进审计，不进 `daily_bars_raw`。

---

## 7. 成员重建与退出

主源：`get_index_weight(['000300.SH'])` 每个 `TRADE_DATE` 的 `CON_CODE`。交叉：`get_index_constituent` 的 `[INDATE, OUTDATE)`。冲突进 `membership_exceptions`，不以区间表覆盖权重表。

规则：每日 300；`date+code` 唯一；权重集合=成员集合；单位检测后和接近 100%；调样日专项；不用最新 300 回填；缺行情/缺行业不删成员；投资池=当日成员；行业全市场股票不得进投资池。

退出（数据层；回测层只消费这些字段）：

1. 第一次不再出现在当日权重表的日期 = `exit_effective_date`。
2. 不从历史库删除该股票。
3. 自该日起 `membership_mask=false`。
4. 不再参与当日截面排名。
5. `known_at` 缺失时：`exit_effective_date` 收盘确认，下一开盘尝试卖出的**政策**写入 schema，本任务不生成持仓。
6. 若当日 `sellable_at_open=false`，记 `UNIVERSE_EXIT_PENDING` 为**事件类型**（审计日志），不是把股票留在投资池。
7. 再入池开新区间，不向前填成员。
8. `INDATE/OUTDATE` 只称 `effective_at`。`known_at=null`，`known_at_source=MISSING_NATIVE_INDEX_ANNOUNCEMENT`。不得 `known_at=INDATE`。

---

## 8. OpenTR 与公司行动

```text
首条有效：CloseTR[t0] = 1
          OpenTR[t0]  = Open[t0] / PreClose[t0]   # 两者均有效且 >0
之后：
OpenTR[t]  = CloseTR[t-1] * Open[t]  / PreClose[t]
CloseTR[t] = CloseTR[t-1] * Close[t] / PreClose[t]
HighTR[t]  = CloseTR[t-1] * High[t]  / PreClose[t]
LowTR[t]   = CloseTR[t-1] * Low[t]   / PreClose[t]
```

- 分红税：跟随交易所 PRECLOSE（税前现金隐含）。`DVD_PER_SHARE_AFTER_TAX_CASH` 只审计。
- 配股：隐含全额认购。
- PRECLOSE 缺失或 ≤0：链中断，不猜测。
- 不把现金股利再乘进收益率。
- `adj_factor` 仅审计。
- 未通过前缀与哨兵测试前，不把复权结果交给 Wingman。
- MCP 首日 `OPEN_TR=1` 不得当正式结果。

公司行动 `known_at` 仅公告字段（`DATE_DVD_ANN`, `ANN_DATE`, `PUBLISH_TIME`, `EXECUTE_DATE` 作为配股**公告**字段时可用）。`CHANGE_DATE` / 除权日不得写入 `known_at`。`known_at_source`：`native` / `derived_conservative` / `missing`。

---

## 9. 缺失行情与方向性掩码

分类：`SUSPENDED | NOT_YET_LISTED | DELISTED | MEMBER_WITH_MISSING_SOURCE_QUOTE | CORPORATE_ACTION_REVIEW | QUERY_FAILED | UNKNOWN`。

停牌 OHLCV 保持缺失。成员行保留。修复写 `repair_log`，不改成员集合，不用隔离库补。

```text
buyable_at_open  = is_member AND has_quote AND not suspended
                   AND open>0 AND volume>0 AND amount>0
                   AND open < high_limited - tick_eps
sellable_at_open = 同上 AND open > low_limited + tick_eps
```

`tick_eps` 默认 0.01。标签的 `entry_blocked(t)` 使用 **t+1** 的 `buyable_at_open`。一字涨停不假设能买；一字跌停不假设能卖。

---

## 10. 退市

仅 `DELISTDATE` / `IS_LISTED`。无处置价。推荐：最后可成交价延迟到第一个 `sellable_at_open`；若直至退市仍不可卖，按完全损失。未确认前快照状态 `DATA_NOT_FORMALLY_VALIDATED`。看实验结果前冻结。

---

## 11. 行业

第一版只申万一级。`get_industry_base_info()` 实盘：511 行；LEVEL_TYPE 1/2/3 = 31/134/346；后缀全 `.SI`；一级样本 `801180.SI`。阶段3修复 MCP 后必须：保留 LEVEL_TYPE==1；用名称与代码确认申万；若混中信则暂停。

历史归属按行业 INDATE/OUTDATE 展开到沪深300成员。冲突则该日行业空，不删股票。Join 前后沪深300行数不变。

v1 必需层：一级指数日行情 + 沪深300历史归属 + 池内 LOO 统计（仅 t 及以前）。不计算三层评分。

增强层默认关闭。一个一级行业样本已有 252 条成分区间。全 A 增强层预估仍为 +12–36 小时、+80–200 GB，须单独确认。

---

## 12. 标签

独立表。信号 t 收盘后，执行 t+1 开盘。

```text
y_1 = log(OpenTR(t+2)/OpenTR(t+1))
y_3 = log(OpenTR(t+4)/OpenTR(t+1))
y_5 = log(OpenTR(t+6)/OpenTR(t+1))
```

保存 signal/entry/exit/label_end、entry_blocked、exit_blocked、label_valid、invalid_reason。未来数据只出现在标签模块。`load_development()` 默认不返回标签。禁止读 Holdout 标签分布。

---

## 13. Development / Holdout

不用机械 10%。推荐：全部完整交易日升序；Holdout 覆盖两个完整日历年且 ≥480 日（精确边界用日历写入 `splits/holdout_dates.parquet` + sha256）；其前为 Development；同一天同一区域；1/3/5 日共用边界；purge 20 日；`splits/HOLDOUT.lock`；修复与阈值不得用 Holdout。

在确认日历边界前，不把 Holdout 日期写进代码常量。

---

## 14. Wingman 面板

新类 `data_pipeline/hs300/panel_manager.py`：`HS300PanelDataManager`。只读冻结快照。禁止单票 Parquet 当面板。禁止时间交集对齐。禁止 `fillna(0)`。

`dates[T], symbols[N], raw_ohlcv[N,5,T], causal_prices[N,4,T], membership_mask[N,T], quote_valid_mask, buyable_mask, sellable_mask, industry_id, features[N,0,T]`。每日 membership true 数验收为 300。缺失为 NaN。标签经 `load_labels`。

N = 权重表历史 CON_CODE 并集（含 2010 起调出股票）。T = 全局交易日，不是个股交集。OpenTR 仅在 `OPENTR_START` 之后有值。

不接到训练环。

---

## 15. 测试计划

A 来源完整性；B 成员 300 / 唯一键 / 权重和 / 调样日 / 行业 join 行数；C 行情不等式与停牌未填；D OpenTR、特征、截面、掩码的前缀不变与未来哨兵；E PRECLOSE 缺口可解释；F 每日至多一个一级、LOO；G 同一 raw 重建哈希一致。

人工抽样：每年 5 日 × 10 股，固定种子；强制含调样、XR/WD、停牌、涨跌停、行业变更、退市或长期停牌。

验证报告路径按原思路第二十三节，写在快照 `validation/`。

---

## 16. 状态机

| 状态 | 条件 |
|---|---|
| `RAW_SNAPSHOT_COMPLETE` | 原始抓取 + 基础结构 |
| `DATA_READY_FOR_DEVELOPMENT` | 因果价格、成员、行业必需层、方向掩码验证通过 |
| `DATA_READY_FOR_FORMAL_RESEARCH` | known_at / available_at / 退市 / 公司行动口径冻结 + 前缀与哨兵 + 不可变快照 + Holdout 锁 |
| `DATA_NOT_FORMALLY_VALIDATED` | 默认 |

能读张量 ≠ 正式研究数据。确认后首次构建预期最多 `DATA_READY_FOR_DEVELOPMENT`（若四项规则已冻结且测试通过）。无中证公告、无状态起点终查前，正式研究状态保持保守。

---

## 17. 预估

与 08-21 相同量级：必需层 2–8 小时、15–40 GB。因权重从 2010 年起多约 3 年成员表，增量不大；K 线仍从 2013。状态若从 2016 才完整，2013–2015 将大量 `MEMBER_WITH_MISSING_SOURCE_QUOTE`，不得删除。

---

## 18. 确认后拟修改文件（尚未改）

MCP：`ad_mcp/server.py` + 新增测试。quant_w1ngman：新增 `data_pipeline/hs300/` 与 `scripts/build_hs300_snapshot.py`、测试。不改 RD-Agent、engine 训练环、`train_file.py` 的 Holdout 评价。

---

## 19. 阶段

3 修行业 MCP + 序列化。4 扫描 status 最早日期并冻结 raw。5 标准化。6 OpenTR。7 掩码。8 申万一级 + 池内 LOO。9 `HS300PanelDataManager` 与张量。10 Development 标签。11 Holdout 日期锁。12 自动测试 + 抽样。13 清单/哈希/状态。

本轮已执行阶段 3–13 的数据层交付。未启动 RD-Agent、未算策略收益、未读 Holdout 表现。

---

## 20. 交付对照（2026-08-25）

快照：`D:/Hulucoding/AmAzing_Data/research_snapshots/csi300_2010_present_v1/`  
状态：`RAW_SNAPSHOT_COMPLETE` + `STANDARDIZED_FROZEN` + `DATA_NOT_FORMALLY_VALIDATED`  
面板门禁：`DATA_GATE_PASSED`（`INDUSTRY_UNMAPPED` 为 WARNING，130222 行）

| 项 | 结果 |
|---|---|
| 1 历史沪深300动态成分 | 完成。4041 个交易日 × 300 = 1,212,300 成员行；历史代码并集 800；成员例外 0 |
| 2 未复权日线 OHLCV/成交额 | 完成。供应商 K 线自 2013-01-04；2010–2012 标 `UNAVAILABLE_FROM_VENDOR`；有效行情 904,293 行自 2014-01-02 |
| 3 因果前缀不变复权参考价 | 完成。OpenTR/HighTR/LowTR/CloseTR 由 PRECLOSE 链重建；对 904,293 条有效行情重算 OpenTR 差异 0；起点 **2014-01-02** |
| 4 公司行动审计 | 完成。48,127 行；`known_at` 仅公告字段 |
| 5 停牌/涨跌停/方向成交 | 完成。买卖掩码分离；2013 年 status 0 行，对应成员日记 `STATUS_UNAVAILABLE_FROM_VENDOR` |
| 6 申万一级与行业统计 | 完成。31 个一级；行业区间 1125；冲突例外 8（同日双一级，区间双方丢弃）；池内 LOO 仅 Development；未映射 130,222 成员日（保留成员、行业空） |
| 7 市场/行业/个股三层评分 | **不做**（已冻结）。张量 `features[N,0,T]` |
| 8 1/3/5 日标签 | Development only：9,606,600 行。Holdout 标签未写 |
| 9 Development / Holdout | Holdout 自 **2024-08-26**，**483** 日，hash `e0e47656…9d8c31`；不是机械最后 10%；`readable_metrics=false` |
| 10 Wingman 面板张量 | 完成。`panel/hs300_tensors.npz`：N=800，T=4041，F=0；每日 membership true=300；缺失为 NaN |

| 对照项 | 现在 |
|---|---|
| 已完成 | 阶段1–13 数据层；MCP 行业修复；SDK 冻结 raw（919 请求）；标准化；标签；张量；checksums |
| 未做（范围外） | RD-Agent、公式筛选、策略收益、Holdout 表现、接入训练环 |
| 原生字段 | 日历、权重、成分区间、K线、状态（2013 空）、基本信息、分红/配股/股本、行业 base/constituent/daily |
| 派生字段 | OpenTR 四价、方向掩码、available_at=21:00 DERIVED_POLICY、行业展开、LOO、Development 标签、Holdout 日期 |
| 测试 | `test_hs300_*` 与 `test_hs300_prefix` 通过；OpenTR 全量重算 mismatch=0；年度抽样 820 行（种子 20260825） |
| 异常 | 成员例外 0；行业冲突 8；行业未映射 130222（WARNING）；2010–2013 行情缺口按供应商缺失保留 |
| 仍缺外部数据 | 中证调整公告 known_at；退市处置价；2010–2013 完整 PRECLOSE/K 线 |
| 数据状态 | `DATA_NOT_FORMALLY_VALIDATED`。允许加载 Development 面板/标签/张量。不允许宣布正式研究结论，不允许读 Holdout 表现 |
| 是否允许 Wingman Development 加载 | **是（数据门禁通过）**。是否允许正式研究结论：**否** |

---

## 21. 正式 v2 快照（2026-08-25）

快照：`D:/Hulucoding/AmAzing_Data/research_snapshots/csi300_2014_present_v2/`  
Raw：junction 到不可变 v1 `raw/`（未重采）。  
状态：`RAW_SNAPSHOT_COMPLETE` + `STANDARDIZED_FROZEN` + `DATA_GATE_PASSED` + `RESEARCH_START=2014-01-02`  
研究状态：`DATA_READY_FOR_DEVELOPMENT`  
面板门禁：`DATA_GATE_PASSED`（`INDUSTRY_UNMAPPED` 为 WARNING，13,845 行；覆盖率 98.22% ≥ 95%）  
schema：`hs300_pit_v2` / `w1ngman_raw_snapshot_v2`

| 项 | 结果 |
|---|---|
| 正式研究起点 | **2014-01-02**（请求起点仍记 2010-01-01）。研究日历 3,074 日；Development 2,591 日 |
| 成员 | Development 777,300 行（2,591×300）；Holdout 密封 144,900 行；合计 922,200 = 3,074×300；成员例外 0 |
| OpenTR | 在含 2013 的全量 K 线上链式计算后切到 2014-01-02，不重置 CloseTR。有效行情 759,681。前缀不变审计 12 只通过 |
| 申万一级 | 冲突保留一条；区间补到下一 INDATE−1。区间 993 行；无 L1 历史例外 4。覆盖率 0.9822。known_at 同日 21:00 有效，未来行 0 |
| 特征 | `panel/hs300_features.npz` 与张量 `features` 均为 **[672, 65, 2591]**，词表顺序与 `FEATURE_NAMES` 一致，`VOCAB_VERSION=v9217a2c0d91a`。存储 NaN；适配器读时才 0 填充 |
| 训练适配器 | `data_pipeline.hs300.adapter.HS300WingmanAdapter`。未接入 Engine / RD-Agent / evaluate_holdout |
| 标签 | Development 6,995,700 行。Holdout 标签未写 |
| Holdout | 自 **2024-08-26**，**483** 日，hash 仍为 `e0e47656433a1ae1351e62a150fadf2716175c0246ed6bc21b566ccb459d8c31`。价格在 `sealed/holdout/`，`readable_prices=false`、`readable_metrics=false`。标准化层无 Holdout 日期 |
| 血缘 | 标准化表含 `schema_version, snapshot_id, data_version, source, fetched_at`。清单 `row_count` 与 `rows` 相等 |
| 审计 | `temporal_leakage_report.json` 通过；`prefix_invariance_report.json` 通过 |

未做：RD-Agent、公式筛选、策略收益、Holdout 表现。v1 目录未改写。
