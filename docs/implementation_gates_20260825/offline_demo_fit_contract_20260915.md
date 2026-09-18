# 联合筛选与拟合：离线演示交接合同

版本：`OFFLINE_DEMO_SCORE_BUNDLE / 1.0`。本合同只规范离线研究端与白名单展示端的交接，不授权读取 Holdout、正式 SOTA 接纳、交易下单或生产建议。

## 1. 本次交付边界

- 研究端：联合特征筛选、受限候选公式提案、训练期拟合、外层分数输出及证据记录。
- 白名单端：由另一条任务实现；研究端不读取用户白名单，不筛选白名单股票、不生成权重。
- 低 IC、收益不足或集中风险警告，不阻断本轮离线流程展示；必须如实显示，不能标记“研究通过”。
- 输入时点、训练与预测隔离、Holdout 隔离、数值与程序错误仍为硬边界，不改为警告。
- 合成数据只能标记 `synthetic`，不能宣称真实股票验证。预先准备的 Development 数据只能标记 `development`。
- 本次不承诺找到“最优”因子或策略；离线提案回放不等于已经调用线上 RD-Agent 服务。
- 原 D21 / S2 正式门禁、历史失败结果、正式 SOTA 接纳和建议接口不变；演示结果不写入正式 SOTA。

## 2. 固定状态

完整写盘后，`manifest.json` 必须保留：

- `schema_version: "1.0"`，`artifact_kind: "OFFLINE_DEMO_SCORE_BUNDLE"`。
- `status: "OFFLINE_DEMO_COMPLETED_NOT_VALIDATED"`。
- `data_role: "development"` 或 `"synthetic"`；`run_id` 是非空运行标识。
- `research_go: false`、`holdout_read: false`、`sota_promoted: false`。
- `production_allowed: false`、`whitelist_applied: false`。

下游不得把“演示完成”转换成“策略已验证”。任何后续白名单过滤结果必须另写产物并继承上述未验证与非生产边界，不覆盖本包。

## 3. 文件接口

每次运行写入一个全新、显式命名的 `offline_demo_*` 目录，建议置于工作区 `experiments/` 下；拒绝覆盖已有目录、旧 Stage 目录，以及名称或解析路径含 Holdout / sealed 的目录。

- `manifest.json`：元数据、四类指纹、文件 SHA-256、原始预测网格行数、成员日覆盖率及状态。它最后写入，是唯一完成标识。
- `scores.parquet`：下述七列的长表；不含收益标签、目标权重、买卖建议或订单。
- `selected_features.json`：本次各折实际拟合所用特征的唯一 ID 汇总；各折选择／拟合项放入模型证据，完整搜索轨迹放在 manifest 的 `selection_runs`。
- `formulas.json`：提案或采用的公式描述；未生成公式可以是空列表，不伪造生成记录。
- `model_evidence.json`：可审计、非可执行 JSON，包括实际估计器、拟合范围、折内选择及模式记录；不保存 pickle。

`fingerprints` 必须具备 `model`、`formula`、`fold`、`data` 四项，均为小写 SHA-256。它们用于绑定实际模型配置与实现／候选公式实现与集合／切分／输入数据；写盘器校验格式，运行器负责根据真实内容计算，不能用占位哈希。

实现入口：`factor_research.demo_artifacts.write_demo_bundle(output_dir, report, scores)`，返回 `manifest`、`scores`、`selected_features`、`formulas`、`model_evidence` 的绝对路径。

## 4. 分数长表

- `date`：上海时区信号对应的交易日，保存为不带时区、无时分秒的日期。
- `code`：股票代码字符串，保留交易所后缀和前导零；下游不得转整数。
- `score`：模型原始评分；不是概率、目标收益或权重，不保证不同折的绝对值可比较。
- `score_available_at`：带时区的模拟历史信号可用时点，磁盘统一保存 UTC；换算上海日期后必须等于 `date`。它表示本次历史重放采用的信息截止时间，不宣称系统当年实际生成过该分数，也不是今天写文件的时间。
- `fold_id`：评分所属外层预测折；键 `(date, code, fold_id)` 必须唯一。
- `is_member`：当时是否在冻结交易池中；不能用今天成员身份替代。
- `signal_valid`：且仅当成员行具有有限评分时为真；非成员评分保持缺失。

写盘器不排序选股，不删除缺失成员日，不填 0。`expected_score_rows` 必须等于运行器在预测前冻结的完整外层评估网格行数，包含无效成员行；覆盖分母是所有评估成员日，不是有效分数行数。写盘器验证数量，但不能替代运行器对原始网格身份与数据指纹的检查。

运行器负责确保模型、筛选和公式仅使用该折允许的历史信息；每个输入的实际可用时间不得晚于信号生成时间。写盘器仅能验证输出时戳的显式时区和日期一致性，不把日期校验冒充未来信息审计。

## 5. 输入与失败处理

- 只允许用户明确准备的、已隔离 Development / synthetic 的 NPZ 或 JSON；不递归扫描数据库，不加载全数据后再过滤 Holdout。
- `validate_demo_input_path` 仅做显式路径与后缀保护，不打开文件。运行器还必须验证声明角色、日期边界、可用时间和内容指纹；文件名带 Development 不是安全证据。
- CLI 的 Development 输入为显式目录中的 `development_manifest.json` 与 `development_arrays.npz`，不自动发现真实数据。manifest 必须明确 `data_role: "development"`、整数 `label_horizon_bars: 5`、股票与特征 ID、信号时间、标签结束时间、特征合同哈希、数组文件名与内外层切分。
- 程序可信固定边界为 `2024-08-26T00:00:00+08:00`。输入自行声明的 `holdout_start` 只能等于或早于此边界，不能后移；全部信号和标签结束时间必须严格早于两者。CLI 在打开 NPZ 数组之前检查这些元数据、标签周期和折切分；失败即停止，不以演示为由放行。
- `label_horizon_bars: 5` 是必填输入合同，仍需上游证明标签确实按 5 个交易日构造；单靠声明不能证明数值来源。程序不读取 Holdout 来验证声明。
- 任何非有限 JSON 指标须由运行器以 `null` 配合原因披露，不得写成 NaN 或悄悄替换为好成绩。
- 写盘或 Parquet 失败时保留不完整目录，不写完成 manifest；禁止覆盖重试，改用新的运行目录。
- 无选中特征、非法时间、伪造“已验证”标记或带交易输出字段的包直接拒绝，不冒充完成。

## 6. 下游读取最小步骤

1. 确认完成 manifest 存在，类型、版本、角色及全部禁止生产标记符合合同。
2. 校验 `scores.parquet` 的 SHA-256、行数和七列 schema；仅把分数作为离线展示输入。
3. 明确选择某个信号日期／外层折；再应用白名单及自身展示逻辑，不把未来日期分数用于历史日期。
4. 原始分数包保持不变；展示页持续说明“离线演示，未验证，不构成交易建议”。

本合同不定义白名单筛选算法、组合权重、收益门槛或正式 SOTA 接纳标准。

## 7. 实际拟合流程与能力限制

1. 每个外层折独立准备允许的历史训练数据，晚到输入保持不可用；先按训练区间清除常数、全缺失、完全重复等基础不可用项，不按单因子低 IC 自动删除。
2. 在该折内层时间验证中，开展有限预算的联合筛选、配对与回删，并评价候选公式。内层反馈可用于下一轮提案，不向提案层提供外层测试标签或测试成绩。
3. 根据该折内层选择，实际拟合固定配置的 LightGBM，在后续外层日期输出分数；不能用外层日期回头选本折特征或公式。H5 标签重叠仍执行 purge / embargo 隔离。
4. 汇总分数、实际筛选与模型证据、公式及预算；只生成本合同的离线分数包，不执行经济有效性裁决、正式 SOTA 接纳或白名单配置。

当前默认每外层折 2 轮提案、每轮 4 个提案槽、每轮最多 96 次组合搜索，最终组最多 5 项；基础候选上限 256，最多 5 个外层折，总机会预算上限 4000。实际参数以运行 manifest 的 `config` 和 `budget` 为准。有限配对和逐步筛选不是全组合穷举，不保证发现所有高阶交互；候选库全部登记也不等于全库全部联合拟合。

如搜索没有选出可靠子集，会保留空的 `selected_features` 折记录，并明确标记 `demo_fallback: true`，另用全部可用基础项拟合演示分数；此回退可以超过 5 项，不是“筛选通过”。汇总文件列出实际拟合项，下游必须结合每折回退标记理解，不得把回退项说成已验证有效因子。

LightGBM 训练实际执行，不是评分占位器；缺少依赖或无法拟合会停止。当前只保存分数与 JSON 证据，**不保存可供后续日期复用的训练模型文件**。因此，白名单端只能消费已有日期分数，不能据此宣称已支持任意新日期的在线预测。

## 8. 本地运行方法

在仓库根目录执行；以下命令只说明使用方式。Python 环境须已具备 NumPy、Pandas、SciPy 和 Parquet 引擎。LightGBM 可安装到仓库隔离目录，不替换全局依赖：

```powershell
D:/anaconda/python.exe -m pip install --target .demo_runtime --no-deps lightgbm==4.7.0
```

本项目已准备 `.demo_runtime` 时无须重复安装。CLI 会仅对当前进程加载该目录。最小演示必须明确选择合成数据；默认提案模式为 `offline_replay`：

```powershell
D:/anaconda/python.exe scripts/run_offline_factor_demo.py --synthetic --output experiments/offline_demo_factors_example_01
```

`--synthetic` 产生明确标注的假数据，不读取真实数据库或 Holdout。`offline_replay` 是本地确定性公式提案，不调用 LLM、不联网；实际 LightGBM 拟合仍会运行。输出目录必须不存在，复跑需换新名字。

对经过独立准备和审计的 Development 包，改用 `--development-bundle "明确的Development目录"`，不能与 `--synthetic` 同时使用。本次文档与程序交付不代表已经准备或运行真实数据。

真正的 RD-Agent 提案路径必须同时显式提供 `--proposal-mode rd_agent --allow-network --rd-model MODEL`，并具备相应服务配置。该模式与本地回放区分记录；默认不会启用，本次未执行。配置三个参数不是读取 Holdout、发送原始数据或生产交易的授权。RD-Agent 负责受限公式提案，LightGBM 负责实际拟合，两者不能混称为同一个模型。

白名单由另一条任务实现：只交接七列 `scores.parquet` 与 `manifest.json`，并可读取附带非执行证据核验来源；本轮不修改其代码、不产生权重或下单建议。

## 9. 本次工程验收（2026-09-15）

- 最新完整样例：`experiments/offline_demo_factors_20260915_r1/`；早期独立样例保留，不覆盖历史结果。
- 使用确定性合成输入与本地公式提案，实际 LightGBM 4.7.0 完成两折拟合并输出 640 条七列评分；不是 65+8 项真实行情实验，也不是收益验证。
- 相关单测与回归：236 通过、1 跳过（Windows 符号链接创建权限）；包含真实 LightGBM 拟合、外层标签不影响当折选择和预测、晚到 1 纳秒保留缺失、固定 Holdout 边界、预算和非生产状态检查。
- `holdout_read=false`、`network_used=false`、`research_go=false`、`sota_promoted=false`、`production_allowed=false`；经济与集中风险验证均为 `NOT_RUN`，没有人为改成通过。
- 真实 Development 数据尚未在这条演示管线中运行；实际远程 RD-Agent 提案尚未调用。后续仅可接入明确准备和审计过的 Development 包。
