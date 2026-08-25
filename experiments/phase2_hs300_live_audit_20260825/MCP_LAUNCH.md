# AmazingData MCP 启动配方（无密钥）

工程根：`D:\Hulucoding\AmAzing_Data`

MCP 服务器（唯一入口）：

`D:\Hulucoding\AmAzing_Data\xysz\xysz\WealthManager\ad_mcp\server.py`

Python：`D:\anaconda\python.exe`

环境变量名：`AD_USERNAME` `AD_PASSWORD` `AD_HOST` `AD_PORT`

说明：

- 用户提供的 `PYTHONPATH=src` 只用于 `tech_cross_section`（在该子项目目录下跑 `python -m tech_cross_section`）。**不要**加到 MCP 进程上。
- MCP `server.py` 会 strip 环境变量。直接调用 `ad.login` 时密码尾部空白会导致 `SystemExit(0)`。
- Cursor 本机 MCP 配置写在用户目录 `C:\Users\Administrator\.cursor\mcp.json`，不进 quant_w1ngman 仓库。
- 本对话若仍看不到 AmazingData 工具，需要在 Cursor Settings → MCP 中启用 `amazingdata` 并重载；本任务阶段2仍等待 CONFIRMATION.md，不因此开始大规模下载。
