# 实验登记 Schema（拟冻结 v1）

状态：`CONFIRMED_FREEZE_20260825`

## 目标

登记全部研究机会，而不只登记冠军。实验记录必须把代码、环境、数据、标签、切分、成本、搜索、组合和门禁绑定到一个不可变 ID。

## 顶层实验记录

```json
{
  "experiment_id": "exp-uuid",
  "parent_experiment_id": null,
  "created_at": "timestamp",
  "stage": 3,
  "status": "REGISTERED|RUNNING|COMPLETE|FAILED|INVALIDATED",
  "hypothesis": "5日相对1日是否改善扣费OOF净Sharpe",
  "code_commit": "git-sha",
  "worktree_diff_sha256": "sha256-or-null",
  "environment_lock_sha256": "sha256",
  "data_snapshot_id": "snapshot-id",
  "data_fingerprint": "sha256",
  "universe_version": "string",
  "industry_version": "string",
  "feature_version": "string",
  "label_protocol": "label-v1",
  "split_protocol": "split-v1",
  "cost_version": "A_SHARE_REFERENCE_COST_V1",
  "portfolio_protocol": "top20_equal_v1",
  "random_seed": 20260812,
  "holdout_accessed": false
}
```

## 建议表

### `experiments`

主键 `experiment_id`；保存上述顶层字段、预注册协议路径、开始/结束时间、失败原因和输出 manifest hash。

### `hypotheses`

`hypothesis_id, parent_id, experiment_id, statement, preregistered_decision_rule, status, created_at`

形成 DAG；Analysis Unit 的下一假设必须以当前假设为 parent，不能只写日志文本。

### `tasks`

`task_id, hypothesis_id, stage, kind, payload_json, status, started_at, completed_at`

### `formula_candidates`

```text
candidate_id, experiment_id, round_index, candidate_index,
prompt_version, llm_model, generator_backend,
raw_response_hash, canonical_formula, formula_hash,
tokens_json, token_count, depth, operator_count,
generation_status, execution_status, elapsed_seconds,
duplicate_of, failure_reason
```

非法、重复、超时和 VM 异常都必须有行。`candidate_index` 使累计搜索机会可审计。

### `validation_results`

```text
result_id, candidate_id, label_id, fold_id,
coverage, invalid_fraction, ic, icir, rank_ic, rank_icir,
net_sharpe, net_return, max_drawdown, turnover,
bootstrap_ci_low, bootstrap_ci_high,
p_raw, p_adjusted, multiple_testing_family, fdr_pass,
industry_coverage, max_industry_contribution,
segment_direction_ratio, payload_json
```

### `sota_events`

`event_id, experiment_id, candidate_id, action, previous_sota_hash, new_sota_hash, marginal_delta, correlation_max, decision_reasons_json, created_at`

每次只接纳一个；接纳后所有剩余候选产生新的验证版本。

### `portfolio_runs`

`run_id, experiment_id, horizon, phase, fold_id, portfolio_protocol, gross_metrics_json, net_metrics_json, cost_breakdown_json, constraints_json, fallback_level, reason_codes_json`

### `gate_results`

`gate_result_id, experiment_id, gate_type, status, evidence_manifest, unresolved_fields_json, created_at`

`gate_type` 只允许 `DATA|STATISTICAL|PORTFOLIO|PRODUCT`，不得合并一条“all_passed”。

### `release_candidates`

`release_id, experiment_id, code_hash, config_hash, data_fingerprint, sota_hash, portfolio_hash, api_schema_hash, status, frozen_at`

### `holdout_access_log`

```text
access_id, release_id, data_fingerprint, holdout_start, holdout_end,
reserved_at, process_identity, purpose, status,
result_manifest_hash, completed_at
```

唯一键：`release_id + data_fingerprint + sota_hash + model_config_hash`。非 `reserved` 行不可更新/删除。失败访问也消耗该 Holdout 使用机会并保留。

### `forward_shadow_runs`

`run_id, generated_at, feature_available_at, data/model/whitelist versions, holdings_hash, recommendation_hash, execution_result, realized_1d, realized_3d, realized_5d, immutable_hash`

## 指纹覆盖

`data_fingerprint` 至少哈希：raw manifest、表 schema、日期和 symbol 顺序、OHLCV/amount/PRECLOSE、成员、行业、公司行动、交易状态、feature masks、split dates。`validation_protocol_hash` 至少哈希：标签、成本、折、purge/embargo、Bootstrap、多重检验、阈值、LightGBM参数、验证代码。

## 不可变规则

1. 注册后阈值字段不可原地修改；变更创建 child experiment。
2. raw/标准化/结果文件用 SHA-256 清单，禁止覆盖。
3. 候选和失败记录不可删除。
4. Holdout结果不能写入 Analysis Unit 或 RD-Agent feedback 表。
5. 任何输出数字若没有 experiment_id 和指纹，只能标记 `UNREGISTERED_EXPLORATORY_RESULT`。

## 当前迁移

现有 `factor_research_store` 表可迁移为只读 legacy namespace；现有单票实验导入后标 `LEGACY_EXPLORATORY_BASELINE`，不写成新 release。当前空的 `multi_symbol` SQLite 不需要数据迁移，但文件仍不覆盖。
