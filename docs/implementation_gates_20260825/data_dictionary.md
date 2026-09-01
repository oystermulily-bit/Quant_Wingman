# 数据字典（已冻结 v1）

状态：`CONFIRMED_FREEZE_20260825`

## 通用规则

- 日期：`date32`，时刻：带 `Asia/Shanghai` 时区的 UTC 可比较时间戳。
- 股票代码统一 `000001.SZ` / `600000.SH`；指数 `000300.SH`。
- 所有标准化表必须有：`schema_version`、`snapshot_id`、`data_version`、`source`、`fetched_at`、`source_row_hash`。
- 事件表必须区分：`event_at`、`effective_at`、`known_at`、`available_at`。
- `*_source` 枚举：`NATIVE`、`DERIVED_POLICY`、`MISSING`。
- 缺失值保持 null/NaN 和原因码；不得用0、ffill或bfill填充 OHLCV、成交额、交易状态或行业归属。
- 标签与特征物理分目录；特征构建代码不得导入标签表。

## 原始请求清单 `raw_request_manifest`

| 字段 | 类型 | 约束 |
|---|---|---|
| request_id | string | 主键，不可变。 |
| snapshot_id | string | 冻结快照版本。 |
| mcp_server_sha256 | string | 64位hex。 |
| sdk_version | string | 必填。 |
| method | string | SDK/MCP方法。 |
| parameters_json | json | 排序序列化后哈希。 |
| requested_at/completed_at | timestamp | 必填。 |
| response_path | string | raw相对路径。 |
| response_sha256 | string | 必填。 |
| row_count | int64 | 非负。 |
| success/retry_count/error | bool/int/string | 失败也保留。 |

## `trading_calendar`

主键：`date`

`date, market, is_trading_day, session_open, session_close, is_complete_session, available_at`

最近完整交易日由该表和当前 Asia/Shanghai 时刻决定，不能只用系统日期。

## `security_master`

主键：`code, valid_from`

`code, name, exchange, list_date, delist_date, security_type, board, st_flag, valid_from, valid_to, known_at`

退市证券不得从历史表删除。

## `universe_membership`

主键：`date, index_code, code`

```text
date, index_code, code, is_member, weight_pct,
effective_at, known_at, known_at_source,
entry_effective_date,
membership_source, source_request_id
```

验收：每个完整交易日 `000300.SH` 恰好300个不同 code；权重集合和成员集合一致；权重和约100%。异常不许通过删行“修复”。每日成员表不得包含尚未实现的 `exit_effective_date`。

## `membership_spell_audit`

主键：`code, spell_id`

`code, spell_id, index_code, entry_effective_date, realized_exit_date, censored_at_snapshot_end`

已实现退出日只写审计表。快照末日仍在指数内的区间 `realized_exit_date` 为空，`censored_at_snapshot_end=true`。

## `execution_bars`

主键：`date, code`

字段与 `daily_bars` 相同，但覆盖**曾入选成分**的全部可报价日，不按当日是否仍是成员切片。回测强制出指数平仓与缺报价估值必须读此表，不得只用成员面板。

## `execution_status`

主键：`date, code`

字段与 `trading_status` 相同，覆盖范围与 `execution_bars` 对齐。

## `industry_membership`

主键：`code, industry_system, level, valid_from`

```text
code, industry_system, level, industry_code, industry_name,
valid_from, valid_to, effective_at, known_at, known_at_source
```

第一版固定申万一级。每只股票每日最多一个一级行业；冲突与未映射进入异常表。

## `daily_bars`

主键：`date, code, revision_id`

```text
date, code,
open_raw, high_raw, low_raw, close_raw, volume, amount, preclose,
open_tr, close_tr, adjustment_type,
available_at, available_at_source,
has_quote, quote_missing_reason, revision_id
```

约束：OHLC>0；`low<=min(open,close)`；`high>=max(open,close)`；volume/amount≥0。停牌或缺行情不得造 K 线。

## `trading_status`

主键：`date, code`

```text
date, code, is_suspended, is_st, is_xr, is_wd,
limit_up_price, limit_down_price,
can_buy_open, can_sell_open,
buy_block_reason, sell_block_reason,
available_at, available_at_source
```

买卖掩码必须分开；`is_tradeable` 单布尔值不足以表达一字涨跌停。

## `corporate_actions`

主键：`action_id`

```text
action_id, code, action_type,
announcement_at, record_date, ex_date, payment_date, effective_at,
cash_dividend, stock_dividend_ratio, rights_ratio, rights_price,
known_at, known_at_source, raw_payload_hash
```

任何派生 known_at 都不得取生效日冒充公告日。OpenTR 构建保存逐日链因子和事件贡献审计。

## `industry_daily_context`

主键：`date, industry_code`

```text
date, industry_code, member_count,
return_1d, return_3d, return_5d, return_20d,
market_excess_1d, market_excess_3d, market_excess_5d,
breadth, volatility_20d, dispersion, average_correlation,
turnover_activity, available_at
```

为个股生成行业特征时另保存 `loo_member_count` 和 Leave-One-Out 结果；目标股票不能影响自己的行业评分。

## `feature_values`

主键：`date, code, feature_id, feature_version`

`date, code, feature_id, value, valid, invalid_reason, feature_available_at, feature_version, input_fingerprint`

张量视图：`X[N,F,T]`，固定 symbol/date 顺序写入 manifest。附加未来日期不得改变历史 value/valid。

## `labels`

主键：`signal_date, code, horizon, label_type, label_version`

```text
signal_date, code, horizon,
entry_date, exit_date, label_start_time, label_end_time,
label_type, gross_log_return, executable_log_return,
entry_blocked, exit_delayed, actual_exit_time,
delisting_return, portfolio_pnl_included
```

`label_type`：`ABSOLUTE`、`MARKET_EXCESS`、`INDUSTRY_EXCESS`。允许未来值只存在于该层。

## `date_splits`

主键：`split_protocol, fold_id, role, date`

`split_protocol, fold_id, role, date, purge_group, holdout_release_id`

`role`：`TRAIN`、`OOF_VALIDATION`、`PURGE`、`EMBARGO`、`HOLDOUT`。同一天全部股票必须同 role。

## 阶段 9 产品表（现在只冻结 schema）

### `user_whitelist`

`whitelist_id, version, created_at, applicable_date, symbols_json, note, data_version, universe_version, industry_version, strategy_version`

### `holdings_snapshot`

`holdings_id, account_id_hash, as_of, code, weight, quantity_nullable, source, created_at`

### `recommendations`

`recommendation_id, schema_version, as_of, generated_at, execution_window, status, reason_codes, model_version, data_version, whitelist_id, weights_json, cash_weight, expires_at`

## 异常/验收输出

`membership_exceptions`、`quote_exceptions`、`industry_exceptions`、`corporate_action_exceptions`、`membership_spell_audit`、`temporal_leakage_report`、`prefix_invariance_report`、`checksums.sha256`。任何硬门槛异常必须保留行级证据。
