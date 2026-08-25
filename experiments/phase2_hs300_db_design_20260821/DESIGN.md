# 沪深300点时研究数据库设计（阶段2）

> 已被 `experiments/phase2_hs300_live_audit_20260825/` 取代。08-25 含 SDK 行数探测：2010 年有权重、无 K 线；PRECLOSE 在 2013-01 样本为空。

状态：`SUPERSEDED_BY_20260825`

审计日：2026-08-21（Asia/Shanghai）。本文件是阶段1审计 + 阶段2设计。未经用户确认，不得进入阶段3以外的下载、转换、标签或训练。阶段3（修复 MCP 行业基础信息）也等确认后再改生产服务器。

本任务范围：数据库、验证报告、Wingman 面板加载接口。不启动 RD-Agent，不筛选公式，不计算策略收益，不读取 Holdout 表现。

---

## 0. 对原思路的总评

方向正确，且与 `docs/01_研究原则.md`、`docs/02_第一期实验协议.md` 的数据合同大部分同向：

- 唯一上游 AmazingData MCP / SDK，不用 MT5。
- 历史成分以每日权重为主，成分区间只做审计。
- 原始响应不可变；禁止 bfill / 当前名单回填 / 用旧 Parquet 补洞。
- OpenTR 用未复权 OHLC + `PRECLOSE`；标签 1/3/5 日与协议公式一致。
- 方向性可交易掩码、缺失分类、公司行动账本、Holdout 审计锁，都应该做。

**不能按原文直接开工。** 有 4 个硬阻塞、若干必须改写的合同冲突。下文推荐方案是“确认后才能执行的设计”，不是已经开始的构建。

---

## 1. 阶段1审计摘要

### 1.1 AmazingData MCP / SDK

| 项 | 结果 |
|---|---|
| 服务器 | `D:/Hulucoding/AmAzing_Data/xysz/xysz/WealthManager/ad_mcp/server.py` |
| SHA-256 | `3BC7F1ADA17436BBDB2CA28FCA91AE7206BA6A2DC5F20158EBB0C1310C370EAD`（与 2026-08-20 预检相同） |
| SDK | AmazingData 1.1.9 |
| 工具数 | 58 |
| 当前 Cursor 会话 | 未注册该 MCP |
| 审计 shell 中 `AD_*` | 未设置；本次未做 2010 实盘探测 |

真实 SDK 方法（不要用 MCP 里写错的名字）：

| 需求 | SDK |
|---|---|
| 交易日历 | `BaseData.get_calendar` |
| 指数权重 | `InfoData.get_index_weight` |
| 指数成分区间 | `InfoData.get_index_constituent` |
| 未复权日线 | `MarketData.query_kline`（`Period.day`） |
| PRECLOSE / 涨跌停 / 停牌 / ST / XR / WD | `InfoData.get_history_stock_status` |
| 基础信息 / 上市退市 | `InfoData.get_stock_basic` |
| 分红 / 配股 / 股本 | `get_dividend` / `get_right_issue` / `get_equity_structure` |
| 公告 | `get_announcement_stock_list` |
| 行业基础 | `InfoData.get_industry_base_info`（**没有** `get_industry_index_info`） |
| 行业成分 / 权重 / 日行情 | `get_industry_constituent` / `get_industry_weight` / `get_industry_daily` |

指数权重字段含 `WEIGHT`，手册标明单位为 **百分比**。检测规则应先验 `日度权重和 ∈ [99, 101]`；仅当明显落在 `[0.99, 1.01]` 时才记为小数。两者都不满足则失败，不得猜测。

`get_index_weight` 支持的代码包括 `000300.SH`。权重表是每日成员主源；`INDATE`/`OUTDATE` 只是生效区间。

### 1.2 MCP 仍须修复（确认后阶段3）

| 工具 | 问题 |
|---|---|
| `mcp_industry_index_info` | 引用不存在的 `end_date` 和 `get_industry_index_info`；必炸 |
| `mcp_industry_index_constituent` | `date` 被忽略；`serialize_dataframe` 可能吃掉 `dict` 返回 |
| `mcp_industry_index_weight` / `quote` | 同样可能把 `dict[str, DataFrame]` 交给 `serialize_dataframe` |
| 若干 ETF/KZZ 工具 | 方法名与 SDK 不符（本任务不用，但说明 MCP 不能当唯一批量通道） |

`mcp_corporate_action_timeline` **没有**把 `known_at` 设成 `INDATE`，这点符合要求。但派生过宽：

- 配股 `KNOWN_AT = max(EXECUTE_DATE, ANN_DATE)`
- 股本 `KNOWN_AT = max(ANN_DATE, CHANGE_DATE)`

`CHANGE_DATE` / 可能的实施日进入 `known_at`，违反“不得把生效日当成公告日”。正式账本必须在快照层重算，MCP 派生值只作对照。

`mcp_total_return_open` 不用 bfill，公式与本稿一致，前缀测试在预检中通过。正式 OpenTR 仍须从**冻结 raw** 重算，禁止训练时现查 MCP。

### 1.3 2010 覆盖（硬阻塞）

手册 2.2.1：

- 股票 / 指数 K 线：**2013 年至今**
- 期货 2010-04：仅中金所，与 A 股日线无关

2016-08-12 旧构建：`requested_start=2013-01-01`，`actual_start=2013-01-04`，3304 个交易日，724 只历史成分。原思路要求 2010-01-01 起且“不得静默缩短”。因此必须列出缺口并等待决定，不能把 `START_DATE` 悄悄改成 2013。

指数权重、行业成分的最早日期手册未单列。即使权重能早于 2013，没有 K 线和 `PRECLOSE` 也无法构造 OpenTR 与标签。2010-01-01 至 2012-12-31 至少对行情是 `UNAVAILABLE_FROM_VENDOR`。

本次未登录 SDK，未用 2010 实盘行数复核手册。确认后第一件探测只记行数与日期。

### 1.4 旧数据隔离

`D:/Hulucoding/AmAzing_Data/output/hs300_daily_2013_present/` 在 2026-08-21 **目录不存在**。构建脚本 `build_hs300_dataset.py` 第 344 行仍是：

```text
data.groupby("code")["adj_factor"].ffill().bfill()
```

`is_tradeable` 把涨跌停当可成交。构建日志：724 只股票、991200 行、缺行情 19531、成员数 min/median/max=300、耗时 1556.6 秒。该规模只允许对照，禁止进入正式库。见 `QUARANTINE.md`。

### 1.5 Wingman 现接口

仓库仍是 MT5 + 单票 Parquet：

- 有 `ParquetDataManager` / `MT5DataManager` / `SingleSymbolDataManager`
- **没有** `HS300PanelDataManager`
- 标签只有 `log(open[t+2]/open[t+1])`，不是 OpenTR，没有 3/5 日
- Holdout 代码是 `chronological_holdout_10_v1`（最后 10% bar）
- `times.py` 对 OHLC 使用 `ffill().bfill()`
- `MT5DataManager` 在交集过短时 `ffill` 再用 `fillna(0.0)`，会把缺失伪装成 0
- 训练路径会调用 Holdout 评价；本任务不得接入这些路径

### 1.6 磁盘与时间

| 盘 | 空闲 |
|---|---|
| C: | ~34 GB（不够做主库） |
| D: | ~180 GB（够必需层；全 A 增强层需另批） |

旧构建仅成分+因子+状态+K线约 26 分钟。新库多公司行动、公告、行业、权重日表、校验，必需层预估 2–8 小时、约 15–40 GB（公告量大则靠上限）。全 A 增强层预估见第 11 节。

### 1.7 最近完整交易日

规则保持：`TIMEZONE=Asia/Shanghai`，若当前是交易日且本地时间 < 15:00，`END_DATE` 为上一完整交易日。审计时刻为 2026-08-21 11:39，若当日开市，则 `END_DATE` 不得为 2026-08-21，应为上一交易日（预期 2026-08-20，须用日历确认）。下载当日再冻结，不在设计里写死。

---

## 2. 与研究协议的冲突（必须改写或改协议）

| 原思路 | 协议 / 宪法 | 设计处理 |
|---|---|---|
| `HOLDOUT_RATIO=0.10` | 协议第6节禁止机械最后 10%；Holdout 至少 2 个日历年、480 交易日；第一期不读表现 | **不用 10%**。约 3310 个交易日的 10% ≈ 331 日，低于 480 |
| 数据库目标含“三层评分” | 三层评分在产品待办 B；第一期行业不是硬门槛 | 本任务只做行业统计与掩码，不算评分 |
| 面板 `features[N,F,T]` | 第一期信号只有 MOM_20 / REV_5 / LOW_VOL_20，且稍后才跑 | v1 面板不预置研究特征；`F=0` 或仅因果价格诊断列 |
| 退出规则 16–24（目标权重、UNIVERSE_EXIT_PENDING） | 本任务不算策略收益 | 只落成员/可卖掩码与政策说明；持仓状态不属于数据库 |
| purge 覆盖 5 日+执行延迟 | 协议推荐 20 日 | 采用 20 日 |
| 每日必须恰好 300 | 旧样本恰好 300，但退市空窗理论上可能 ≠300 | 硬验收 + 例外表；不删行凑数 |

单票 `000001` 已看过的尾段不得复用为沪深300 Holdout。日历切分的沪深300 尾段只要本项目从未读过其组合表现，仍可用；本任务仍不得计算该段收益。

---

## 3. 确认后的推荐合同

见 `CONFIRMATION.md`。推荐冻结值：

```text
INDEX_CODE              = 000300.SH
REQUESTED_START         = 2010-01-01
ACTUAL_START            = 2013-01-04   # 待日历确认
END_DATE                = latest_completed_trading_day
COVERAGE_GAP            = 2010-01-01 .. 2012-12-31  UNAVAILABLE_FROM_VENDOR
FREQUENCY               = D1
TIMEZONE                = Asia/Shanghai
INDUSTRY                = 申万一级 only（LEVEL_TYPE=1，分类系统须在探测后核实）
ENHANCEMENT_LAYER       = OFF in v1
OPEN_TR                 = PRECLOSE 链，税前隐含，全额配股
INDEX_KNOWN_AT          = null / MISSING_NATIVE_INDEX_ANNOUNCEMENT
AVAILABLE_AT            = event_date 21:00 Asia/Shanghai, DERIVED_POLICY
DELISTING               = delayed last-trade then 100% loss if never sellable
HOLDOUT                 = last ~2 calendar years, >=480 days, purge 20
SNAPSHOT_ROOT           = D:/Hulucoding/AmAzing_Data/research_snapshots/csi300_2013_present_v1/
STATUS_AFTER_BUILD      = DATA_NOT_FORMALLY_VALIDATED until rules frozen AND tests pass
```

若用户坚持 2010 完整覆盖：停工，不缩短，不补洞。

---

## 4. 采集清单与批次

唯一上游：修复后的 AmazingData SDK（MCP 同源）。训练只读快照。

所有研究拉取使用手册 4.4 **参数组2**：`begin_date` + `end_date`，`is_local` 无效。禁止依赖默认本地 HDF5 缓存当正式快照。`fetched_at` 必须是本次请求完成时间。

### 4.1 查询顺序

1. `get_calendar(market='SH')` → 交易日、`END_DATE`
2. `get_index_weight(['000300.SH'])` 按年切块
3. `get_index_constituent(['000300.SH'])` 全历史区间
4. 由权重表得到历史 `CON_CODE` 并集（不得因缺行情删除）
5. `get_stock_basic(code_union)`
6. 按年、每批 20–40 只：`query_kline`、`get_history_stock_status`
7. 按批：`get_dividend`、`get_right_issue`、`get_equity_structure`
8. 公告：可选压缩范围（见下）
9. `get_industry_base_info()` 一次
10. 过滤申万一级 `INDEX_CODE` 后：`get_industry_constituent`、`get_industry_daily`；HS300 映射用成分区间展开，不需要全市场日权重
11. 复权因子 `get_backward_factor` / `get_adj_factor` **仅审计**，不进特征

### 4.2 批次与断点

| 接口 | 代码批 | 日期窗 |
|---|---|---|
| index_weight | 1 只指数 | 1 年 |
| kline / status | 20–40 | 1–2 年 |
| dividend / rights / equity | 20–40 | 全样本或 3 年 |
| industry_daily | 全部一级代码一次 | 1–2 年 |
| announcements | 10–20 | 1 年 |

每个请求写入：

```text
raw/<method>/year=YYYY/part=<batch_id>.parquet
raw/<method>/year=YYYY/part=<batch_id>.meta.json
```

成功文件永不覆盖。重试写 `part=<batch_id>.retry<n>.*`。`.partial` 写完校验行数后再 rename。

公告全量可能主导耗时。v1 推荐：只拉与公司行动日期 ±20 交易日重叠的公告，全量公告标为可选。需确认。

### 4.3 禁止

- 训练中调 MCP
- 用 `is_local=True` 读旧缓存当正式 raw
- 用 `mcp_total_return_open` 代替冻结 raw 上的本地 OpenTR
- 用 MCP 错误行业工具而不先修复

---

## 5. 目录

```text
D:/Hulucoding/AmAzing_Data/research_snapshots/csi300_2013_present_v1/
├─ raw/ ...
├─ standardized/
├─ derived/
├─ labels/
├─ splits/
├─ validation/
├─ panel/                      # 可选物化张量，非唯一接口
├─ manifest.json
├─ schema.json
├─ checksums.sha256
└─ STATUS.txt                  # DATA_NOT_FORMALLY_VALIDATED | ...
```

仓库内只放代码与实验文档，不把数十 GB raw 提交到 git。

每个 raw 响应 meta：

```json
{
  "snapshot_id": "csi300_2013_present_v1",
  "source": "AmazingData",
  "mcp_server_sha256": "3BC7F1ADA17436BBDB2CA28FCA91AE7206BA6A2DC5F20158EBB0C1310C370EAD",
  "sdk_version": "1.1.9",
  "method": "InfoData.get_index_weight",
  "mcp_tool": "mcp_index_constituent_weight",
  "parameters": {},
  "requested_at": "",
  "completed_at": "",
  "response_sha256": "",
  "row_count": 0,
  "success": true,
  "retry_count": 0
}
```

---

## 6. 标准化 Schema（v1）

所有表：`schema_version, data_version, source, fetched_at, snapshot_id`。

| 表 | 主键 | 要点 |
|---|---|---|
| trading_calendar | date | SH 日历；`is_complete_session` |
| universe_membership | date, code | 来自每日权重；`is_member`；`exit_effective_date` 不删历史行 |
| index_weights | date, code | `weight_raw`, `weight_unit`, `weight_pct` |
| daily_bars_raw | date, code | 未复权 OHLCV + amount；禁止填充停牌 |
| stock_status | date, code | PRECLOSE, HIGH/LOW_LIMITED, IS_SUSP, IS_ST, IS_XR, IS_WD |
| stock_master | code | 上市/退市、名称；点时字段另表 |
| corporate_actions | code, event_id | 见第8节 |
| adjusted_price_causal | date, code | OpenTR/HighTR/LowTR/CloseTR + 输入比率 + `REFERENCE_SOURCE=PRECLOSE_CHAIN` |
| tradability_directional | date, code | 第9节 |
| delisting_events | code | 第10节 |
| industry_base | industry_code | 申万一级；`classification_system` |
| industry_membership | date, code | 最多一个一级；失败保留成员、行业空 |
| industry_daily | date, industry_code | 指数行情 |
| industry_statistics | date, industry_code | 仅 t 及以前；个股侧 LOO |
| missing_quote_class | date, code | 第7节 |
| repair_log | repair_id | 不得改变成员集合 |
| audit_index_known_at | date | 恒为 missing native |

代码格式：`000001.SZ` / `600000.SH`。指数只用 `000300.SH`。

允许 ffill 的字段（须在 schema 写理由，同股、向未来、上市前保持缺失、有最大长度）：

- 行业归属（成分区间展开，不是价格 ffill）
- 证券名称等主数据状态

禁止 ffill：OHLCV、amount、PRECLOSE、涨跌停价、停牌虚构行情、adj_factor、成员资格、OpenTR。

---

## 7. 成员重建

主源：`get_index_weight(['000300.SH'])` 的每个 `TRADE_DATE` 上 `CON_CODE` 集合。

交叉：`get_index_constituent` 的 `[INDATE, OUTDATE)`。冲突进 `membership_exceptions`，**不自动以区间表覆盖权重表**。

规则：

1. 验收：每个交易日成员数 = 300；否则例外表 + 不得升正式研究状态。
2. `date+code` 唯一。
3. 当日权重建股票集合 = 成员集合。
4. 权重单位检测见上，标准化后和接近 100%。
5. 调样日前后专项（成员对称差、权重和）。
6. 不用最新 300 回填。
7. 缺行情不删成员。
8. 缺行业不删成员。
9. 投资池 = 当日成员。
10. 行业全市场股票不得进投资池。
11. 第一次权重表不再出现的日期 = `exit_effective_date`；此后 `membership_mask=false`。
12. 再入池开新区间，不向前填成员。
13. `INDATE/OUTDATE` 只称 `effective_at`。
14. `known_at` 空，`known_at_source=MISSING_NATIVE_INDEX_ANNOUNCEMENT`。

无公告时的可执行退出政策（供日后回测，本任务不跑）：`exit_effective_date` 收盘确认，下一开盘尝试卖出；停牌/跌停/无报价则延迟到第一个 `sellable_at_open`。数据库只提供掩码，不生成持仓状态机。

---

## 8. OpenTR 与公司行动

正式公式（与预检 MCP 工具相同，但从 raw 重算）：

```text
首条有效：CloseTR[t0] = 1, OpenTR[t0] = Open[t0] / PreClose[t0]  （若 Open 与 PreClose 均有效）
之后：
OpenTR[t]  = CloseTR[t-1] * Open[t]  / PreClose[t]
CloseTR[t] = CloseTR[t-1] * Close[t] / PreClose[t]
HighTR[t]  = CloseTR[t-1] * High[t]  / PreClose[t]
LowTR[t]   = CloseTR[t-1] * Low[t]   / PreClose[t]
```

冻结假设：

- 分红税：跟随交易所 `PRECLOSE`（税前现金股利隐含）。`DVD_PER_SHARE_AFTER_TAX_CASH` 只审计。
- 配股：隐含全额认购。
- `PRECLOSE` 缺失或 ≤0：该日链中断，`label_input_valid=false`，不猜测。
- 不把现金股利再乘进收益率（避免与 PRECLOSE 双计）。
- 原始 `adj_factor` 仅审计。
- 未通过前缀与哨兵测试前，不把复权结果交给 Wingman。

前缀测试定义：`build(full)[:k] == build(prefix_to_k)`，截断的是**右端未来**，不是改股票首日。水平值因首日归一化，在同一全局起点下可比较。

未来哨兵：把截断日之后的 OHLC 换成极端值，截断日及以前的 OpenTR/特征/掩码必须不变。

公司行动账本：

| 字段 | 规则 |
|---|---|
| known_at | 仅公告字段：`DATE_DVD_ANN`, `ANN_DATE`, `PUBLISH_TIME` 等 |
| known_at_source | `native` / `derived_conservative` / `missing` |
| effective_at / ex_date | `DATE_EX`, `EX_DIVIDEND_DATE`, `CHANGE_DATE` 等，不写入 known_at |
| timing_valid | known_at ≤ effective_at（在日期精度内）；否则例外 |

禁止把 MCP 的 `max(ANN_DATE, CHANGE_DATE)` 当正式 known_at。

普通日 `PRECLOSE ≈ CLOSE[t-1]`；缺口必须能对应 XR/WD/配股/股本/数据异常之一。

---

## 9. 缺失行情与方向性掩码

分类：`SUSPENDED | NOT_YET_LISTED | DELISTED | MEMBER_WITH_MISSING_SOURCE_QUOTE | CORPORATE_ACTION_REVIEW | QUERY_FAILED | UNKNOWN`。

停牌 OHLCV 保持缺失。成员行保留。修复写 `repair_log`，不改成员集合，不用隔离库补。

废弃单一 `is_tradeable` 作为最终成交判断。

```text
signal_valid      = 成员且用于生成信号的输入齐全（不要求 t+1 可买）
buyable_at_open   = is_member AND has_quote AND not suspended
                    AND open>0 AND volume>0 AND amount>0
                    AND open < high_limited - tick_eps
sellable_at_open  = 同上 AND open > low_limited + tick_eps
```

`tick_eps`：A 股默认 0.01，比较前先按交易所精度四舍五入。`HIGH_LIMITED` 已含 10%/20% 制度差异。

含义：开盘已触及涨停则不假设能买；开盘已触及跌停则不假设能卖。一字涨停是子集。

面板上 `buyable_mask[i, date]` 表示 **该日开盘** 能否买，不是信号日。标签的 `entry_blocked(t)` 使用 `t+1` 的 `buyable_at_open`。

---

## 10. 退市

AmazingData 仅 `DELISTDATE` / `IS_LISTED`，无处置价、无现金补偿。

`delisting_events` 保存能拿到的字段 + `disposition_source` + `evidence_quality=vendor_date_only`。

候选（须选一个冻结）：

1. 外部真实处置数据（现无）
2. 最后可成交价延迟退出
3. 预注册保守损失率（须给数字）
4. 无可靠信息按完全损失

推荐：2 然后 4。未确认前状态 `DATA_NOT_FORMALLY_VALIDATED`。

---

## 11. 行业

第一版只申万一级。不得混中信。

`get_industry_base_info()` **没有** `index_type`。MCP 的 `SW/ZX` 参数对应不存在的方法。阶段3修复为调用 `get_industry_base_info`，然后：

1. 保留 `LEVEL_TYPE==1`
2. 用 `INDEX_CODE` / 名称探测是否申万（常见 `801xxx.SI` 或申万2021编码）
3. 若混有中信：过滤失败则暂停，不静默混用
4. `classification_system='SW'` 写入 industry_base；不确定则 `UNKNOWN` 并暂停

历史归属按行业 `INDATE/OUTDATE` 按日展开到沪深300成员。冲突（同日多一级）进例外表，该日行业空，**不删股票**。行业 join 前后沪深300行数必须相等。

### 必需层（v1）

- 申万一级指数日行情
- 沪深300股票历史行业归属
- 沪深300内部宽度、离散度、排名（LOO，仅 t 及以前）

### 可选增强层（默认关闭）

- 申万一级全部历史成分（近全 A）
- 全 A 宽度 / 活跃度 / 波动 / 离散度

预估（数量级，非实测）：

| 项 | 必需层 | 增强层 |
|---|---|---|
| 股票 | ~724 历史沪深300 | ~5000–6000 全 A 历史 |
| 行业指数 | ~28–31 一级 | 同左 |
| 下载时间 | 2–8 小时 | 另加 12–36 小时 |
| 磁盘 | 15–40 GB | 另加 80–200 GB |
| D: 180 GB | 可行 | 勉强，需确认后再拉 |

非沪深300股票只作环境参照，禁止进入 `membership_mask` 与投资池。

行业统计至少包括原思路第十六节列表；个股行业统计 Leave-One-Out。不在本任务输出“行业评分”或“个股评分”。

---

## 12. 标签

独立表 `labels/labels_horizon.parquet`，不回写特征表。

信号 t 日收盘后；执行 t+1 开盘。

```text
y_1(i,t) = log(OpenTR(i,t+2) / OpenTR(i,t+1))
y_3(i,t) = log(OpenTR(i,t+4) / OpenTR(i,t+1))
y_5(i,t) = log(OpenTR(i,t+6) / OpenTR(i,t+1))
```

保存：`signal_date, entry_date, exit_date, label_end_date, entry_blocked, exit_blocked, label_valid, invalid_reason`。

未来数据只出现在标签模块。禁止：标签回填特征、参与缺失填充、行业统计、公式输入、LLM、Holdout 调规则。

无法入场：`entry_blocked=true`，标签可 `label_valid=false`，样本保留。无法退出：不删样本，记 `exit_blocked`。调出日训练样本保留。

`HS300PanelDataManager.load_development()` 默认不返回标签张量。`load_labels(split=...)` 显式调用。Holdout 标签可落盘，本任务禁止读取其分布或表现。

---

## 13. Development / Holdout

不用 `HOLDOUT_RATIO=0.10`（除非用户改写协议）。

推荐生成方式：

1. 取 `ACTUAL_START .. END_DATE` 全部完整交易日，升序。
2. Holdout = 最后两个完整日历年且不少于 480 个交易日（精确边界用日历计算后写入 `splits/holdout_dates.parquet` + sha256）。
3. 其前为 Development。
4. 同一天所有股票同一区域。
5. 1/3/5 日共用边界。
6. purge 20 个交易日：Development 最后可训练信号日满足 `label_end_date` 早于 Holdout 首日的 `available_at`。
7. 只保存日期与哈希。设置 `splits/HOLDOUT.lock`。
8. 缺失修复、行业参数、标准化、阈值不得使用 Holdout。

在用户确认日历边界前，不把 Holdout 日期写进代码常量。

---

## 14. Wingman 面板接口

新类 `data_pipeline/hs300/panel_manager.py`：`HS300PanelDataManager`。

只读冻结快照。禁止 `ParquetDataManager` 读多股票长表当面板。禁止时间交集对齐。禁止 `fillna(0)` 伪装价格。

推荐输出：

| 数组 | 形状 | 说明 |
|---|---|---|
| dates | [T] | 全局交易日，不是个股交集 |
| symbols | [N] | 2013 至今历史成员并集 |
| raw_ohlcv | [N, 5, T] | OHLCV；缺失为 NaN |
| amount | [N, T] | |
| causal_prices | [N, 4, T] | OHLC TR |
| membership_mask | [N, T] | 每日 true 数验收为 300 |
| quote_valid_mask | [N, T] | |
| buyable_mask / sellable_mask | [N, T] | 该日开盘 |
| industry_id | [N, T] | 无映射为 -1 |
| features | [N, 0, T] | v1 空；算子必须支持 mask |

标签经 `load_labels`。非成员不得进当日截面。不得对非成员阶段无限制 ffill。

不把此类接到 `W1ngmanEngine` / RD-Agent / `evaluate_holdout`。

---

## 15. 测试计划

自动化，不允许只看几行。

A. 来源：每请求有 raw、参数、时间、sha256；成功响应不可覆盖；重复构建哈希一致。  
B. 成员：每日 300、唯一键、权重集合、权重和、行业 join 行数不变、调样日。  
C. 行情：OHLC 不等式、正价格、非负量额、非交易日无日线、缺失有原因、停牌未填。  
D. 未来信息：OpenTR、价格/市场/行业特征、截面排名、掩码的前缀不变；未来哨兵。  
E. 公司行动：PRECLOSE 缺口可解释。  
F. 行业：每日至多一个一级；无映射/冲突单独报告；LOO；前缀不变。  
G. 可重复性：同一 raw 构建两次，行数/Schema/排序/sha256/指纹一致。

人工抽样：每年 5 日 × 10 股，固定种子；强制含调样、XR/WD、停牌、涨跌停、行业变更、退市或长期停牌。

验证报告路径按原思路第二十三节，写在快照 `validation/`。

---

## 16. 状态机

| 状态 | 条件 |
|---|---|
| `RAW_SNAPSHOT_COMPLETE` | 原始抓取 + 基础结构 |
| `DATA_READY_FOR_DEVELOPMENT` | 因果价格、成员、行业必需层、方向掩码验证通过 |
| `DATA_READY_FOR_FORMAL_RESEARCH` | known_at / available_at / 退市 / 公司行动口径冻结 + 前缀与哨兵通过 + 不可变快照 + Holdout 锁 |
| `DATA_NOT_FORMALLY_VALIDATED` | 默认；任何未决规则或失败测试 |

能读张量或能启动 Wingman ≠ 正式研究数据。

确认后首次构建的预期状态：`DATA_NOT_FORMALLY_VALIDATED` 或最多 `DATA_READY_FOR_DEVELOPMENT`（若用户已冻结 4 项规则且测试通过）。在中证公告接入前，正式研究状态仍应保守。

---

## 17. 预估数据量与耗时

基于 2026-08-12 日志外推（724 只、3304 日）：

| 阶段 | 时间 | 磁盘 |
|---|---|---|
| 日历 + 权重 + 成分 | 10–30 分钟 | <1 GB |
| K线 + 状态 | 1–3 小时 | 5–10 GB raw |
| 公司行动 | 30–90 分钟 | <1 GB |
| 公告（压缩） | 1–3 小时 | 5–15 GB |
| 行业必需层 | 20–60 分钟 | <2 GB |
| 标准化 + OpenTR + 掩码 + 标签 + 校验 | 1–2 小时 | 5–10 GB |
| **必需层合计** | **2–8 小时** | **15–40 GB** |
| 增强层 | +12–36 小时 | +80–200 GB |

C: 34 GB 不足。快照放 D:。

---

## 18. 确认后拟修改文件（尚未改）

MCP（阶段3）：

- `D:/Hulucoding/AmAzing_Data/xysz/xysz/WealthManager/ad_mcp/server.py`
- 对应 MCP 测试（若无则在 `ad_mcp/tests/` 新增）

quant_w1ngman（阶段4+）：

- 新增 `data_pipeline/hs300/`（fetch, raw_store, membership, open_tr, tradability, industry, labels, splits, validate, panel_manager）
- 新增 `scripts/build_hs300_snapshot.py`
- 新增 `tests/unit/test_hs300_*.py`、`tests/property/test_hs300_prefix.py`
- **不改** `model_core/rd_agent_generator.py`、`model_core/engine.py` 训练环、`train_file.py` 的 Holdout 评价

明确不在本任务修改：用 `HS300PanelDataManager` 替换 MT5 训练默认路径。

---

## 19. 阶段计划（确认后）

3. 修 MCP 行业基础信息 + 序列化，测试。  
4. 小探测 2010/2013 实际最早日期（行数 only），然后下载并冻结 raw。  
5. 标准化日历、成员、权重、行情、状态。  
6. 公司行动账本 + OpenTR。  
7. 方向掩码 + 退市表。  
8. 申万一级归属 + 必需层统计。  
9. `HS300PanelDataManager`。  
10. 1/3/5 日标签（独立表）。  
11. Development/Holdout 日期清单，不读表现。  
12. 全量自动测试 + 抽样报告。  
13. 清单、哈希、异常、状态。

现在停在阶段2。

---

## 20. 最终交付对照（构建完成后填写）

构建完成后必须如实列出原思路第二十六节的 1–10 项，以及成员退出规则 11–24 中属于数据层的实现情况。本设计阶段预先声明：

| 项 | 现在 |
|---|---|
| 已完成 | 阶段1审计、阶段2设计、隔离说明 |
| 未完成 | 阶段3–13 全部 |
| 原生字段 | 见第1.1节 SDK 列 |
| 派生字段 | OpenTR、掩码、available_at 政策、行业日展开、标签、Holdout 日期 |
| 待冻结人为规则 | 见 CONFIRMATION.md |
| 仍缺外部数据 | 中证调整公告 known_at；退市处置价/现金补偿 |
| 测试结果 | 未跑正式构建测试 |
| 异常数量 | N/A |
| 数据状态 | 尚无快照；设计状态 `STAGE_2_COMPLETE_AWAITING_CONFIRMATION` |
| 是否允许 Wingman Development | **否** |
