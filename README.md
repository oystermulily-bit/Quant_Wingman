# quant_w1ngman

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
