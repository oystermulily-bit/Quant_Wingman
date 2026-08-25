# 项目地图

快照基准：Git `a010326564156f0ce6eae77d29f1e366d1c1eede`，分支 `main`。审计开始时工作树已有用户修改：`docs/01_研究原则.md`、`docs/02_第一期实验协议.md`、`docs/03_产品待办.md`，以及未跟踪的预检/设计目录；本次未覆盖这些内容。

## 运行架构

```text
单票 Parquet / MT5 兼容数据
→ data_pipeline
→ model_core.features [N,65,T]
→ RD-Agent（硅基流动 DeepSeek）提出 RPN token
→ StackVM [N,T]
→ 旧收益评分 + FactorResearchLoop
→ ValidationUnit + 固定 LightGBM + SOTA/Knowledge Forest
→ best_{symbol}.json
→ 旧连续多空回测 / 实时监控
→ FastAPI + 静态 HTML/CSS/JS
```

拟议的沪深300点时面板、市场/行业/个股三层职责、白名单和 A 股长仓组合目前不在上述运行链中。

## 目录职责

| 路径 | 当前职责 | 阶段判断 |
|---|---|---|
| `model_core/` | 65 特征、62 算子、StackVM、RD-Agent 生成器、训练引擎、旧回测、10% Holdout | 保留公式语言与 RD 生成；标签、面板和组合语义需新路径实现。 |
| `factor_research/` | Validation Unit、固定 LightGBM、保留策略、SOTA、知识森林、一次性 release 行登记 | 可复用基础设施；需加入多重检验、完整搜索登记和沪深300协议。 |
| `data_pipeline/` | MT5、多品种交集/并集对齐、单票 Parquet 加载 | 兼容保留；不得作为正式沪深300面板加载器。 |
| `strategy_manager/` | MT5 连续多空信号、仓位状态、风控与运行器 | 旧产品兼容；不得复用为 A 股目标权重。 |
| `execution/` | 旧执行配置和价格源 | 不进入第一期沪深300研究。 |
| `web/` | FastAPI、训练/回测/实时任务管理和静态 UI | 研究工具保留；正式建议 API/页面推迟到阶段 9。 |
| `backtest_viz/` | 旧回测统计、图表和报告 | 工程回归可用；正式组合回测需按新合同实现。 |
| `tests/` | 281 项通过、17 项跳过的单元/属性/烟雾测试 | 证明现有工程回归，不证明正式研究有效。 |
| `data/` | 单票本地数据与紫金矿业样例 | 历史资产只读，不作为正式沪深300数据库。 |
| `checkpoints/` | 旧单票训练检查点 | 不迁移到正式沪深300研究；只作兼容/归档。 |
| `strategies/` | 旧单票冠军和 Holdout JSON | 兼容读取；不得当作正式参考组合。 |
| `factor_research_store/` | SQLite + NPZ 因子研究存储 | 当前 `multi_symbol` 数据库为空。 |
| `experiments/` | 单票114轮快照、MCP预检、沪深300数据库设计 | 证据资产；不得覆盖。 |
| `docs/` | 研究原则、第一期协议、产品待办及本门禁包 | 设计权威；现有三份文档有用户未提交修改。 |
| `scripts/` | 下载、E2E、教程和训练 I/O 辅助脚本 | 按用途逐项保留，不自动纳入正式研究。 |
| `lord/` | LoRD 旧实验辅助 | 公式生成已禁用旧 REINFORCE/LoRD 路径；兼容保留。 |
| `paper/` | 论文/参考材料 | 非运行依赖。 |

## 根入口分类

| 类别 | 文件 | 处理 |
|---|---|---|
| Web 启动 | `run_web.py`、`start_web.bat`、`_launch_web.bat` | 保持兼容；README 当前误写 `start_web.py`，需在产品阶段修正。 |
| 公式训练 | `train_file.py`、`train_single.py`、`main.py` | 旧单票/多品种研发入口；正式面板另建显式入口。 |
| 回测 | `run_backtest.py`、`run_v1_backtest.py`、`backtest_*.py` | 旧工程回归；不得混作新组合回测。 |
| MT5/实盘 | `run.py`、`live_trade.py`、`monitor_live_risk.py`、`train_ftmo*.py` | 沪深300第一期排除，暂留兼容。 |
| 数据下载/检查 | `download_*.py`、`fetch_*.py`、`check_*.py`、`verify_*.py` | 旧来源工具；正式数据只读冻结 MCP 快照。 |
| 分析工具 | `analyze_*.py`、`deep_*.py`、`scan_all_factors.py`、`trace_index_factor.py` | 辅助/旧研究，不自动进入门禁。 |

## 外部资产

- AmazingData MCP：`D:/Hulucoding/AmAzing_Data/xysz/xysz/WealthManager/ad_mcp/server.py`
- MCP SHA-256：`3BC7F1ADA17436BBDB2CA28FCA91AE7206BA6A2DC5F20158EBB0C1310C370EAD`
- AmazingData SDK：现有审计记录为 `1.1.9`
- 开发手册：`D:/Hulucoding/AmAzing_Data/AmazingData开发手册.pdf`
- 旧构建脚本：`D:/Hulucoding/AmAzing_Data/build_hs300_dataset.py`
- 旧输出目录当前不存在；即使恢复也只能 `QUARANTINED_REFERENCE_ONLY`。

## 目标新增边界（确认后）

```text
data_research/
├─ snapshot_manifest.py
├─ hs300_panel.py
├─ temporal_contract.py
├─ corporate_actions.py
├─ labels.py
├─ splits.py
└─ gates/data_gate.py

research_protocol/
├─ horizon_baseline.py
├─ statistical_gate.py
└─ experiment_registry.py

portfolio_research/
├─ constraints.py
├─ optimizer.py
├─ execution_masks.py
└─ gates/portfolio_gate.py
```

目录名是阶段 2 设计建议，不代表已创建实现。
