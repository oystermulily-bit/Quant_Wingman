"""CSI 300 point-in-time snapshot configuration (v2)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

RAW_SNAPSHOT_ID = "csi300_2010_present_v1"
SNAPSHOT_ID = "csi300_2014_present_v2"
_SNAPSHOT_BASE = Path("D:/Hulucoding/AmAzing_Data/research_snapshots")
RAW_SNAPSHOT_ROOT = _SNAPSHOT_BASE / RAW_SNAPSHOT_ID
SNAPSHOT_ROOT = _SNAPSHOT_BASE / SNAPSHOT_ID
MCP_SERVER_PATH = Path(
    "D:/Hulucoding/AmAzing_Data/xysz/xysz/WealthManager/ad_mcp/server.py"
)
INDEX_CODE = "000300.SH"
# 原始抓取曾从2010年发起，用于保留供应商覆盖缺口和来源谱系；
# 它不是模型允许使用的研究样本起点。
REQUESTED_START = date(2010, 1, 1)

# 2026-08-28冻结：正式研究、Development切分和数据门禁统一从
# 第一个具备OpenTR与方向性交易状态的完整交易日开始。
RESEARCH_START = date(2014, 1, 2)
TIMEZONE = "Asia/Shanghai"
FREQUENCY = "D1"
SCHEMA_VERSION = "hs300_pit_v2"
BATCH_SIZE = 30
KLINE_BATCH_YEARS = 1
WEIGHT_BATCH_YEARS = 1
STATUS_AFTER_RAW = "RAW_SNAPSHOT_IN_PROGRESS"
ENHANCEMENT_LAYER = False
INDUSTRY_COVERAGE_MIN = 0.95
FEATURE_WARMUP_BARS = 60
SEAL_CONFIRM_PHRASE = "UNSEAL_HOLDOUT_PRICES"
