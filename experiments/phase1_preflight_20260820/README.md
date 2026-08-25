# 第一期实验预检记录

结论：`MODEL_NOT_VALIDATED`

本次只执行了协议门禁检查，没有启动正式的 1 日 / 3 日 / 5 日比较，没有计算收益、夏普或 Bootstrap，也没有读取或划定后的封存 Holdout。

## 阻塞原因

1. 已修复 `D:/Hulucoding/AmAzing_Data/xysz/xysz/WealthManager/ad_mcp/server.py` 的环境变量、序列化、资源 URI 和 stdio 问题；58 个工具均可注册，关键工具真实只读查询通过。但当前 Codex 会话仍需在客户端配置后重启，且研究数据合同仍有下列阻塞项。
2. 项目仓库内只有历史单票数据，不能替代沪深300点时面板。
3. 发现的本地候选数据集 `D:/Hulucoding/AmAzing_Data/output/hs300_daily_2013_present` 来自 AmazingData Python SDK，但缺少协议要求的 `known_at`、`available_at`、公司行动事件表、可审计 `Open_tr`、退市处置规则和完整 manifest。
4. 该候选数据的构建脚本对复权因子使用 `ffill().bfill()`；未来因子可能回填历史缺失，不能直接通过前缀不变性门禁。
5. 候选数据把涨跌停继续视为可交易，没有买入/卖出方向性的执行阻塞规则。
6. 协议第11节的九项冻结选择尚未写入实验记录。

## MCP能力复核

- `known_at`：未找到。指数成分股只有 `INDATE/OUTDATE`，没有调整公告时间。
- `available_at`：未找到原生字段。日线和历史状态只有交易日期，可在本地采用保守规则派生，但必须写入协议并测试。
- 公司行动时态：部分具备。分红、配股、股本和公告接口包含公告日、生效日、除权除息日、`PUBLISH_TIME`、`PRECLOSE`、`IS_XR_SEC/IS_WD_SEC`。
- `Open_tr`：新增了基于原始 `OPEN/CLOSE/PRECLOSE` 的因果构造工具；离线属性测试和真实数据前缀测试均通过。正式实验仍须把原始响应、公司行动账本、抓取时间、版本和哈希冻结到快照。
- 退市处置：只有 `DELISTDATE/IS_LISTED`；未发现退市处置价格或现金补偿字段。

本次能力查询只记录接口字段名、行数和Schema，没有输出原始行情值。

详细字段、文件哈希和 Schema 审计结果见 `preflight_report.json`；MCP 修复与测试证据见 `mcp_repair_report.json`。

## 恢复正式实验所需输入

- 在客户端注册修复后的 AmazingData MCP 并重启会话；服务端加载与工具注册已经验证。
- 补充成分调整公告时间来源；若确实无法取得，必须明确修改研究原则，不能把 `INDATE/OUTDATE` 自动宣称为完整 `known_at`。
- 把分红、配股、股本、公告、历史状态与 `Open_tr` 结果冻结为不可变事件账本和快照，并补齐 manifest。
- 为退市处置价格补充来源，或在看结果前冻结保守损失规则。
- 冻结 Development / Holdout 日期、公司行动和退市规则、交易成本与流动性门槛。
- 确认协议推荐的简单信号、Top 20 等权、统计阈值和 Bootstrap/Holm 设置。

在这些条件满足前，任何使用当前候选数据得到的“3日或5日优于1日”结论都不属于正式第一期实验结果。
