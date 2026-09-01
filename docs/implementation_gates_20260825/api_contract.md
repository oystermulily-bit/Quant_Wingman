# API 核心合同（拟冻结 v2）

状态：`CONFIRMED_FREEZE_20260825`  
实施阶段：阶段9。当前旧 API 继续作为研究工具，不具备正式建议资格。

## 1. 兼容原则

- 不删除或静默改变现有 `/api/health`、配置、训练、回测、实时监控路径。
- 新合同统一 `/api/v2`，所有响应带 `schema_version`。
- 旧策略缺新字段时标记 `LEGACY_RESEARCH_ARTIFACT`，不得推断为验证通过。
- 新 API 不返回 API Key、密码或完整本机敏感路径。
- 研究结果、公式集成和用户建议分开，不能把公式 `best_score` 当建议置信度。

## 2. 新路径

| 方法/路径 | 用途 | 最早阶段 |
|---|---|---|
| `GET /api/v2/system-status` | 四类门禁、整体状态、版本 | 阶段3 |
| `GET /api/v2/data-status` | 快照、覆盖、时态、数据新鲜度 | 阶段3 |
| `GET /api/v2/research/experiments/{id}` | 预注册协议、OOF和门禁报告 | 阶段3 |
| `GET /api/v2/research/factors` | 公式/SOTA研究结果，不含建议 | 阶段5 |
| `GET/POST /api/v2/whitelists` | 用户白名单版本 | 阶段9 |
| `POST /api/v2/holdings` | 当前持仓快照 | 阶段9 |
| `POST /api/v2/portfolio/recommendation` | 下一调仓日目标权重 | 阶段9且Holdout后；KEEP_1D 允许拒绝骨架（空仓位、现金 100%、`MODEL_NOT_VALIDATED`），不得夹带可交易权重 |
| `GET /api/v2/releases/{id}` | 冻结模型/数据/代码/审计状态 | 阶段8 |

## 3. 通用响应头部

```json
{
  "schema_version": "w1ngman_api_v2",
  "request_id": "uuid",
  "generated_at": "2026-08-25T21:05:00+08:00",
  "status": "MODEL_NOT_VALIDATED",
  "reason_codes": ["DATA_GATE_FAILED"],
  "data_version": null,
  "model_version": null
}
```

HTTP状态与业务状态分开：研究未通过是合法业务响应，可用 HTTP 200 + `MODEL_NOT_VALIDATED`；参数缺失用 400/422；内部异常用500 + `INTERNAL_ERROR`。

## 4. 正式建议请求

```json
{
  "as_of": "2026-08-25",
  "whitelist_id": "manual_pool_001:v3",
  "holdings_id": "portfolio_001:20260825",
  "account_value": null,
  "requested_horizon": 5
}
```

## 5. 正式建议响应

```json
{
  "schema_version": "w1ngman_recommendation_v2",
  "status": "MODEL_NOT_VALIDATED",
  "reason_codes": ["STATISTICAL_GATE_NOT_PASSED"],
  "as_of": "2026-08-25",
  "data_available_at": "2026-08-25T21:00:00+08:00",
  "generated_at": "2026-08-25T21:05:00+08:00",
  "execution_window": "NEXT_TRADING_DAY_OPEN",
  "expires_at": "2026-08-26T09:30:00+08:00",
  "horizon": 5,
  "unit": "ACCOUNT_WEIGHT",
  "data_version": null,
  "universe_version": null,
  "industry_version": null,
  "model_version": null,
  "strategy_version": null,
  "whitelist_id": "manual_pool_001:v3",
  "market_risk_budget": null,
  "cash_weight": 1.0,
  "expected_turnover": 0.0,
  "positions": []
}
```

门禁不通过时不得夹带看似可执行的目标股票列表。

## 6. 业务状态枚举

产品响应：

`OK, NO_TRADE, INSUFFICIENT_EVIDENCE, STALE_DATA, UNTRADABLE, TARGET_VALID_BUT_EXECUTION_BLOCKED, MISSING_WHITELIST, MISSING_HOLDINGS, MODEL_NOT_VALIDATED, INTERNAL_ERROR`

系统/发布状态：

`DATA_NOT_FORMALLY_VALIDATED, DATA_READY_FOR_DEVELOPMENT, RESEARCH_GATE_FAILED, KEEP_1D_BASELINE, KEEP_SIMPLE_MODEL, FROZEN_RELEASE_CANDIDATE, HOLDOUT_FAILED, MODEL_NOT_VALIDATED, FORWARD_TEST_ONLY, PRODUCTION_ELIGIBLE`

组合内部状态见 `portfolio_contract.md`，通过 reason code 映射到产品状态。

## 7. 状态优先级

```text
INTERNAL_ERROR
→ DATA_NOT_FORMALLY_VALIDATED / MODEL_NOT_VALIDATED
→ STALE_DATA
→ MISSING_WHITELIST
→ MISSING_HOLDINGS（仍可只返回目标权重时作为reason而非硬失败）
→ INSUFFICIENT_EVIDENCE / NO_TRADE
→ TARGET_VALID_BUT_EXECUTION_BLOCKED
→ OK
```

`PRODUCTION_ELIGIBLE` 是允许评估 `OK` 的必要条件，不是每次都必须交易。

## 8. 数据新鲜度

比较 `data_available_at`、最近完整交易日、当前时间和策略有效期。不是最近完整交易日的数据必须 `STALE_DATA`，不得复用旧建议。数据刚收盘但未到冻结 `available_at` 也不得生成。

## 9. 旧字段映射

| 旧字段 | 新含义 |
|---|---|
| `formula` / `formula_decoded` | 只进入 research factor endpoint。 |
| `best_score` | `legacy_research_score`，不进入用户建议。 |
| `generator_backend` | 研究元数据。 |
| `data_file` | legacy artifact locator；正式版本使用 `data_version`/snapshot hash。 |
| `holdout_report` | 旧单票审计引用；不能自动成为 release audit。 |

## 10. 产品门禁测试

覆盖：旧路由兼容、新 schema 校验、密钥不回传、全部状态枚举、STALE_DATA、模型未验证拒绝、白名单/持仓缺失、执行阻塞、公式报告与建议隔离、最小启动与主要 API。
