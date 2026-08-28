# 阶段2确认清单（2026-08-25）

当前状态：`CONFIRMED_AND_EXECUTED`  
快照状态：v2 `DATA_READY_FOR_DEVELOPMENT` / `DATA_GATE_PASSED`（`csi300_2014_present_v2`）  
v1 目录保持只读。面板门禁：`DATA_GATE_PASSED`

用户已启用 MCP 并按推荐口径执行阶段 3–13 数据层。凭证不会写入仓库。密码必须 `strip`。

以下条目在构建中按推荐值冻结，不再随结果修改。

未确认前不得开始：MCP 大规模修复以外的下载、标准化、标签、面板物化或训练。

回复方式：对下列每一项写 `同意推荐` / `改成……` / `暂缓`。

账号已用于行数探测。密码含尾部空白时 SDK 会 `SystemExit(0)`，后续正式采集必须 `strip`。凭证不会写入仓库。

## 必须先决定（阻塞开工）

1. **日期起点**
   - 推荐：`REQUESTED_START=2010-01-01` 写入清单，不静默缩短。
   - 成员权重实盘下界：2010-01-04（2010-01 每日 300 只）。
   - K 线实盘下界：2013-01-04（2010-01 与 2012-12 为 0 行）。
   - 2010-01-01 至 2012-12-31 的股票日线 / 状态标为 `UNAVAILABLE_FROM_VENDOR`。
   - 不得用旧 Parquet、当前名单或权重表 `CLOSE` 补 OHLCV。
   - 备选：停工，直到另接能覆盖 2010 年股票日线的数据源。

2. **OpenTR / PRECLOSE 起点（新硬阻塞）**
   - 实盘：`000001.SZ` 的 `get_history_stock_status` 在 2013-01 为 0 行，2016-01 为 20 行。
   - 推荐：确认后阶段4先做状态接口的按年行数扫描（≥2 只历史成分），再开始大规模 K 线下载。
   - `OPENTR_START = max(KLINE_START, STATUS_START)`。在查清前不得宣称 2013 年起可做因果价格。
   - 备选：若你有文档证明状态从 2013 起完整，再指定复核代码列表。

3. **Holdout**
   - 推荐：按 `docs/02_第一期实验协议.md` 用日历切分，不用机械最后 10%。
   - 窗口：覆盖两个完整日历年且不少于 480 个交易日；purge = 20；精确日用交易日历冻结。
   - 以 2013–2026-08-24 计，10% ≈ 330 日，低于 480。
   - 备选：坚持 `HOLDOUT_RATIO=0.10`，则必须同时改写实验协议第6节。

4. **指数 known_at**
   - 推荐：`known_at=null`，`known_at_source=MISSING_NATIVE_INDEX_ANNOUNCEMENT`。成员以每日权重 `TRADE_DATE` 为 `effective_at`。
   - 不得令 `known_at=INDATE`。成分区间 INDATE 最早 2005-04-08，仍只是生效日。
   - 备选：另行接入中证指数历史调整公告后再升级正式研究状态。

5. **available_at**
   - 推荐：`event_date` 当天 21:00 Asia/Shanghai，`available_at_source=DERIVED_POLICY`；真实 `fetched_at` 另存。
   - 备选：下一交易日 08:00。
   - 禁止把派生值写成数据源事实。

6. **退市**
   - 推荐：最后可成交价延迟到第一个 `sellable_at_open=true`；若直至退市仍不可卖且无处置价，按完全损失。
   - 在看到任何实验结果前冻结。未冻结时 `DATA_NOT_FORMALLY_VALIDATED`。

7. **分红税与配股**
   - 推荐：OpenTR 只跟随交易所 PRECLOSE；税前现金股利、全额认购配股。首日 `CloseTR=1`，`OpenTR=Open/PreClose`。不要用 MCP 的首日 `OPEN_TR=1`。
   - 税后口径另开版本。

8. **申万一级增强层（全A成分）**
   - 推荐：v1 **不下载**。必需层：31 个一级（实盘 LEVEL_TYPE=1）指数行情 + 沪深300历史归属 + 池内 Leave-One-Out。
   - 须在修复 MCP 后核验 31 个一级名称均为申万、无中信。样本代码 `801180.SI`，后缀全 `.SI`。
   - 全 A 增强层预估另加 12–36 小时、80–200 GB。

9. **三层评分**
   - 推荐：本任务只落库统计，不计算市场/行业/个股评分，不写入 `features`。

10. **每日必须恰好 300 只**
    - 推荐：硬验收。2010-01 与 2013-01 样本已满足每日 300、权重和 ≈100%。任何交易日 ≠300 写入例外表，不得升到 `DATA_READY_FOR_FORMAL_RESEARCH`，除非书面接受例外。不得为凑 300 增删成员。

11. **快照根目录**
    - 推荐：`D:/Hulucoding/AmAzing_Data/research_snapshots/csi300_2010_present_v1/`
    - 目录名表示请求起点，不表示 2010 年行情已齐。C: 仅 33.3 GB。D: 约 179 GB。

12. **采集通道**
    - 推荐：quant_w1ngman 内 SDK 批量客户端；`is_local=False` + `begin_date/end_date`；密码 strip；捕获 `SystemExit`。
    - 不在训练中调用 MCP。不把 `mcp_total_return_open` 当正式 OpenTR。

13. **END_DATE**
    - 本次探测日 2026-08-25 为交易日且 15:00 前，故最近完整交易日 = **2026-08-24**。
    - 推荐：开工下载当天按同一规则重算并冻结。

## 确认后才执行的下一动作

1. 修复 MCP `mcp_industry_index_info`（改为 `get_industry_base_info`）及行业序列化，并补测试。
2. 小流量扫描 `get_history_stock_status` 实际最早非空日期（只记行数与日期）。
3. 核验 31 个一级行业均为申万。
4. 按冻结规则下载并冻结原始响应。
