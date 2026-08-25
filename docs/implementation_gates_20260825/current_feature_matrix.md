# 当前特征矩阵与模型维度

## 运行时维度

| 对象 | 当前形状 | 含义 |
|---|---|---|
| 单个 OHLCV 字段 | `[N,T]` | N 个品种、T 个共同时间点；单票时 `N=1`。 |
| 原始输入集合 | 5 个 `[N,T]` + time | `open/high/low/close/volume`；没有行业、点时成分和交易掩码。 |
| 特征张量 | `[N,65,T]` | 运行时词表精确为 65 个特征。 |
| 公式 token | `[L]` | RPN 表达式；当前词表总长 127，算子偏移 65。 |
| 公式输出 | `[N,T]` | StackVM 标准化后的因子序列。 |
| 当前目标 | `[N,T]` | `y_1[t]=log(open[t+2]/open[t+1])`，末两位填 0。 |
| 旧目标仓位 | `[N,T]` | `tanh(factor)`，范围 `[-1,1]`。 |

运行时事实：`vocab_version=v9217a2c0d91a`、65 特征、62 算子、词表大小 127。`model_core/config.py` 中 `INPUT_DIM` 的注释“==10”和 `model_core/features.py` 中“现有30特征”的注释已经过时，不能作为真实维度依据。

## 65 个现有特征

```text
RET, RET5, RET20, MA_DIFF, SLOPE20,
ATR, RVOL, HL_RANGE, VOL_REGIME,
DEV, DEV60, RSI14, PRESSURE, AC1,
VOL_RATIO, VOL_Z, PV_CORR,
REL_RET5, REL_RET20, REL_VOL,
VWAP_DEV, BOLL_POS, BOLL_WIDTH, MACD_HIST, OBV_SLOPE,
MFI14, WILLR_14, CCI_14, ROC_12, TYPICAL_DEV,
EMA_RATIO_12_26, TREND_STRENGTH_50, PRICE_POS_50,
TRIX_15, PPO, ULT_OSC, RET_ACCEL,
GK_VOL, PARKINSON_VOL, YANG_ZHANG_VOL, RS_VOL,
AMIHUD_ILLIQ, KYLE_LAMBDA, CMF_20, AD_LINE_SLOPE,
STOCH_K_14, STOCH_D_3, AROON_OSC_25, DMI_ADX_14,
DMI_DIFF_14, TRIX_SIGNAL, DONCHIAN_POS_20, KELTNER_POS_20,
ICHIMOKU_KIJUN_DEV, ICHIMOKU_TENKAN_DEV, SUPERTREND_DIR,
SAR_DIST, ROLL_SKEW_20, ROLL_KURT_20, HURST_50,
FRACTAL_DIM_30, AC2, RET_ENTROPY_20,
CS_RANK_RET5, CS_ZSCORE_RET20
```

## 当前可复用部分

- 大多数滚动特征使用历史窗口，已有前缀不变性属性测试。
- StackVM 输出标准化已经移除基于全序列标准差的路径分支；`tests/unit/test_vm_causality.py` 已覆盖。
- VM 在数值清洗前保存 `invalid_mask`，Validation Unit 可计算真实覆盖率。
- Validation Unit 的 RankIC 已使用平均秩处理并列值。

## 不能直接进入沪深300正式研究的部分

1. `CS_RANK_RET5` 使用双 `argsort`，没有平均处理并列值；停牌/相同收益较多时产生任意顺序。
2. 单票时两个截面特征退化成时序归一化，因此单票公式含义与真正横截面公式不同。
3. 特征出口把 NaN/Inf 清为 0；正式面板必须同时保存覆盖掩码，不能只看清洗值。
4. 当前没有 `is_member`、`industry_code`、`known_at`、`available_at`、公司行动、方向性可交易掩码。
5. 当前没有市场、行业 Leave-One-Out 和个股残差特征。
6. `data_pipeline/data_manager.py` 在交集太短时会对 OHLCV `ffill` 并把起始缺失填 0，不符合正式面板数据合同。
7. 当前目标只支持 1 日，且 engine 的 legacy IC 又把已前视的 `target_ret` 额外移动一位，和 PnL 标签不一致。

## 阶段 3 最小矩阵（确认后）

```text
daily_bars / masks / membership / industry
→ X_price[N,F_price,T]
→ X_market[1,F_market,T]
→ X_industry[G,F_industry,T]
→ X_stock_residual[N,F_residual,T]
→ y_abs[N,3,T]           # H=1,3,5
→ y_market_excess[N,3,T]
→ y_industry_excess[N,3,T]
```

第一期先用预注册简单信号验证周期，不启动 RD-Agent、LightGBM 或复杂三层权重。
