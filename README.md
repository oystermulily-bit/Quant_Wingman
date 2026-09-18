# quant_w1ngman

## 股票选择与配置流程（2026-09-16：离线流程已实现）

目标产品输出当期沪深300全部300只股票，按最终公式当期得分排序，由用户选择形成白名单后生成配置。当前**独立离线流程**已接入样本外收益校准、协方差风险与成本优化，所选股票推荐目标均为正；包含版本化白名单、配置证据、建议区间、现金、执行模拟、回放复核框架与后续观察记录。缺分成员保留展示，选入后报告不可行，不进行Top-N截断。

运行 `python run_portfolio.py --port 8766` 后打开 [离线股票配置](http://127.0.0.1:8766/portfolio)；主训练页也有同名入口。详见[新版实现、API与验收说明](docs/implementation_gates_20260825/portfolio_optimizer_implementation_20260917.md)。默认数据为300只合成示例，真实市场五折全相位验证尚未完成。

当前默认300只和16只包均为**合成历史回放**，评分来自 `model_prediction`联合模型原始输出，不是单公式收益率、IC或账户权重。尚无真实沪深300评分包、新日期推理或正式投资建议。既有Top20实验保留原口径，正式建议与Holdout仍受验证门禁控制。

## 当前沪深300研究路线（2026-09-15）

用户已批准[阶段4/5联合研究S2](docs/implementation_gates_20260825/joint_research_s2_20260915.md)及程序修改：已有审核特征直接参与内层联合筛选/拟合，外层按时间验证完整流程，不要求原三个信号先通过，也不按单因子低IC自动淘汰。当前为 `WORKFLOW_APPROVED / EXPERIMENT_FREEZE_PENDING`；候选、日期、最终信号输出、统一预算和组合风险合同仍须冻结。

本轮工程提供独立外层预测与准入/预算基础设施，不代表真实经济研究、SOTA接纳或生产已完成。首版不动态生成公式或启动RD-Agent；未来受限公式扩展也只能使用内层反馈并共享预算。模型预测与等权必须显式选择，不能默认为同一策略。

旧D21-v1为 `KEEP_1D_BASELINE`、v2为 `INSUFFICIENT_EVIDENCE`、v3为 `D21_V3_FAILED`，原产物保留。系统仍是 `MODEL_NOT_VALIDATED`，正式交易建议及Holdout继续关闭。下文通用/兼容公式引擎能力不等于当前沪深300正式路径已经获得运行资格；沪深300Holdout以冻结合同的 `max(10%, 480交易日)` 且至少两个日历年为准，不使用下文兼容引擎的机械10%说明。

## 项目简介

quant_w1ngman 是一个本地量化因子研究、验证、回测与实时监控项目。RD-Agent 通过硅基流动 DeepSeek 接口提出受约束的公式 token，本地 StackVM 执行公式；行情、收益和 Holdout 数据不会发送给 LLM。

## 主要功能

- RD-Agent 因子假设和 RPN 公式提案；
- 本地公式合法性检查与因子计算；
- Walk-Forward 样本外验证、相关性去冗余和扣费组合评价；
- 固定 LightGBM 对 SOTA 因子矩阵执行边际增量检验；
- SQLite 知识森林记录假设、实验、失败原因和分析报告；
- 最后 10% 数据强制作为独立 Holdout，不参与选优或 LLM 反馈；
- 策略回测、训练曲线和实时行情监控。

## 技术栈

- Python 3.10+、PyTorch、NumPy、Pandas、PyArrow；
- LightGBM 固定非线性因子评价器；
- FastAPI、Uvicorn、原生 HTML/CSS/JavaScript；
- SQLite 元数据与 NPZ 因子矩阵。

## 目录结构

```text
factor_research/   SOTA存储、标准验证、LightGBM、保留策略、知识森林和分析单元
model_core/        特征、算子、StackVM、RD-Agent生成器、引擎和底层回测
data_pipeline/     Parquet及行情数据读取和特征工程
strategy_manager/  策略信号与实时执行
web/               FastAPI服务和静态网页
tests/             单元、集成和界面测试
```

## 安装依赖

```powershell
cd C:\path\to\quant_w1ngman
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 启动命令

```powershell
.\.venv\Scripts\python.exe start_web.py
```

浏览器访问 `http://127.0.0.1:8765/`。

## API Key 配置方式

密钥不会写入 `web_settings.json`。推荐在启动服务前通过 PowerShell 环境变量配置：

```powershell
$env:SILICONFLOW_API_KEY="你的Key"
$env:SILICONFLOW_MODEL="deepseek-ai/DeepSeek-V4-Pro"
```

网页中临时输入的 AI Key 或通知密钥只在当前服务进程内有效，重启后需重新配置。训练断点只能导入网页导出的 v2 `.zip` 安全训练包；不再接受可触发 pickle 反序列化的 `.pt` 上传。

## 训练、验证和回测流程

```text
LLM提出假设和公式token
→ StackVM本地计算
→ Walk-Forward标准验证
→ 与SOTA去冗余
→ 固定LightGBM OOF增量检验
→ 本地硬门槛接纳或拒绝
→ 聚合指标交给Analysis Unit
→ 下一轮
```

公式只有通过覆盖率、稳定 IC/RankIC、相关性、固定 LightGBM 增量和扣费组合门槛后才进入 SOTA。研发结束后，冻结冠军公式可在最后 10% Holdout 上执行一次独立审计；该结果不回流选优。回测继续读取兼容的 `strategies/best_{symbol}.json`。

## 实时监控功能

网页可为多个品种配置策略，显示本地信号、方向、行情状态，并按现有配置发送通知。实时监控不会修改训练因子或 SOTA 数据。

## UI 主题说明

界面采用黑、白、紫主题，保留训练、回测和实时监控三个工作区。

## 常见问题

- **提示缺少 LightGBM**：重新执行依赖安装命令，不能用其他模型替代固定基线。
- **旧检查点无法续训**：Holdout 协议现为 `chronological_holdout_10_v1`，旧协议检查点为防止数据边界混用会被拒绝。
- **一轮没有因子入库**：查看 `factor_research_store/<symbol>/research.sqlite3` 中的失败原因；这表示候选未通过硬门槛，不是程序卡死。
- **没有策略文件**：只有至少一个因子进入 SOTA 后才会产生可供下游使用的冠军公式。

## 许可证说明

本项目沿用仓库内现有许可证及第三方依赖各自许可证。量化研究和回测结果不构成投资建议。
