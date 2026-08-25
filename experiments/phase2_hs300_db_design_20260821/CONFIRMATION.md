# 阶段2确认清单

当前状态：`STAGE_2_COMPLETE_AWAITING_CONFIRMATION`

未确认前不得开始：MCP 大规模修复以外的下载、标准化、标签、面板物化或训练。

回复方式：对下列每一项写 `同意推荐` / `改成……` / `暂缓`。

## 必须先决定（阻塞开工）

1. **日期起点**
   - 推荐：`REQUESTED_START=2010-01-01`，`ACTUAL_START=2013-01-04`（手册股票/指数K线起点；2010-01-01 至 2012-12-31 标为 `UNAVAILABLE_FROM_VENDOR`）。
   - 不得用旧 Parquet 或当前名单补 2010–2012。
   - 备选：停工，直到另接能覆盖 2010 的数据源。

2. **Holdout**
   - 推荐：按 `docs/02_第一期实验协议.md` 用日历切分，不用机械最后 10%。
   - 推荐窗口（待用交易日历冻结精确日）：Development = `ACTUAL_START` 至 Holdout 开始前一日；Holdout = 最近约两个完整日历年（不少于 480 个交易日）；purge = 20 个交易日。
   - 备选：坚持 `HOLDOUT_RATIO=0.10`，则必须同时改写实验协议第6节，否则数据合同与研究宪法冲突。

3. **指数 known_at**
   - 推荐：`known_at=null`，`known_at_source=MISSING_NATIVE_INDEX_ANNOUNCEMENT`。成员资格以每日权重表 `TRADE_DATE` 为 `effective_at`。
   - 不得令 `known_at=INDATE`。
   - 备选：另行接入中证指数历史调整公告后再升级状态。

4. **available_at**
   - 推荐：`event_date 21:00 Asia/Shanghai`，`available_at_source=DERIVED_POLICY`；真实 `fetched_at` 另存。
   - 该规则允许 t 日收盘后生成信号、t+1 开盘执行。
   - 备选：下一交易日 08:00。
   - 禁止把派生值写成数据源事实。

5. **退市**
   - 推荐：规则 2 优先（最后可成交价，延迟到第一个 `sellable_at_open=true`）；若直至退市仍不可卖且无处置价，则规则 4（按完全损失）。
   - 在看到任何实验结果前冻结。
   - 未冻结时快照状态保持 `DATA_NOT_FORMALLY_VALIDATED`。

6. **分红税与配股认购**
   - 推荐：OpenTR 只跟随交易所 `PRECLOSE` 隐含调整；等价于税前现金股利、全额认购配股。
   - 公司行动账本只审计 `CLOSE[t-1]` 与 `PRECLOSE[t]` 的缺口，不再把股利叠加入 OpenTR。
   - 税后口径另开版本，不混进 v1。

7. **申万一级增强层（全A成分）**
   - 推荐：v1 **不下载**。只做必需层：申万一级行业指数日行情 + 沪深300历史行业归属 + 沪深300内部行业统计（Leave-One-Out）。
   - 全A增强层预估见 DESIGN.md，需单独确认。

8. **三层评分**
   - 推荐：本任务只落库统计，不计算市场/行业/个股评分，不把评分写入 `features`。
   - 三层评分属于产品待办 B，第一期周期比较之前不应冻结评分公式。

9. **每日必须恰好 300 只**
   - 推荐：作为硬验收；任何交易日成员数 ≠ 300 写入 `membership_exceptions.parquet`，快照不得升到 `DATA_READY_FOR_FORMAL_RESEARCH`，除非用户书面接受例外。
   - 不得为凑 300 删除或添加成员。

10. **快照根目录**
    - 推荐：`D:/Hulucoding/AmAzing_Data/research_snapshots/csi300_2013_present_v1/`
    - C: 仅剩约 34 GB，不能作为主存储。D: 约 180 GB 空闲，够必需层。

11. **采集通道**
    - 推荐：quant_w1ngman 内 SDK 批量客户端；方法名与 MCP 工具对应；记录 `sdk_version`、`mcp_server_sha256`；`is_local=False` 且只用 `begin_date/end_date` 参数组。
    - 不在训练中调用 MCP。不把 `mcp_total_return_open` 的即时结果当作正式 OpenTR。

## 确认后才执行的下一动作

1. 修复 MCP `mcp_industry_index_info`（改为 `get_industry_base_info`）及行业序列化，并补测试。
2. 小流量探测：2010 与 2013 的权重/K线/行业实际最早日期（只记行数与日期，不落行情值）。
3. 按冻结规则下载并冻结原始响应。
