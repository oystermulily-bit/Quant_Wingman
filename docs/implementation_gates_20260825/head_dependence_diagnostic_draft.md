# 头部依赖诊断草案

状态：`DRAFT_NOT_FROZEN`  
系统状态：仍为 `MODEL_NOT_VALIDATED`  
范围：Development OOF only；不读 Holdout；不修改 Stage 4 已冻结的 GO 阈值

本文回答的问题是：

> 新增因子后，Wingman 在沪深300里选出 Top-20；Stage 4 删掉贡献最大的头部股票后，因子对剩余股票变弱甚至给出负收益。这在研究上应如何分类、如何诊断、如何处置？

这是诊断协议草案，不是已经通过的因子结论，也不是软件故障报告。

---

## 1. 思路：先分类，再检验

“删头部后失效”在文献里不是一种病。至少有四类机制会给出同一个表面现象：

| 代码 | 机制 | 典型表现 | 处置 |
|---|---|---|---|
| `NAME_IDIOSYNCRATIC` | 少数名字的极端观测撑起溢价 | 洛伦兹曲线极陡；行业分散 | 拒绝入库 |
| `INDUSTRY_PROXY` | 因子其实是行业均值/行业动量 | 行内残差 IC≈0 或为负；申万 LOO 翻号 | 迁到行业层，禁止再加进个股 Alpha |
| `THEME_HEAD` | 跨官方行业的主题社群（白酒、新能源、半导体） | 单一主题贡献过半；Leave-One-Theme-Out 翻号 | 最多当该主题的环境开关 |
| `CONSTRUCTION_ONLY` | Top-20 浓缩倾斜的伪影 | 全截面弱，只有头部组合好看 | 拒绝；不靠优化器硬凑 |

对应你们看到的句子——“因子描述的是单行业股票，删掉头部后对其他股票效果变弱甚至负收益”——标准落点是 `INDUSTRY_PROXY` 或 `THEME_HEAD`，不是“因子算错了”。

为什么会变负：若分数 ≈ 行业哑变量，头部是该行业里标签最大的几只；删掉它们后，剩余样本里该行业只剩跟风股，因子仍在给整个行业加分，而跟风股的条件期望往往低于截面均值，RankIC 就会翻号。

---

## 2. 现有 Stage 4 检查测到了什么，缺什么

现行代码在 `research_stage4/runner.py` 的 `_concentration_ok`：

1. 用 `simple_ensemble * label` 对每只股票求 OOF 均值，丢掉贡献最大的约 5% 名字，再看剩余 RankIC 是否与总体同号。
2. 按申万一级做 Leave-One-Out，看 RankIC 增量是否翻号。

这已经抓住了红旗，但有三个协议问题：

1. **识别与评价没有隔离。** 用事后 `score × 标签` 选头部，再用同一套标签评 IC，接近 in-sample 修剪。Knez and Ready (1997) 的修剪实验本身就是在揭示这种脆弱性；它适合当敏感性，不适合当正式 GO。
2. **IC 头部 ≠ 组合头部。** RankIC 在 300 只股票上计算；参考组合每天只持有 20 只。两者必须分开报。
3. **申万一级 ≠ 主题簇。** A 股白酒、新能源、半导体经常跨子行业同涨。只做行业 LOO 会漏掉社群头部。

正式参考组合继续：Top-20、等权 5%、不递补第 21 名、头部不可交易时变现金。诊断可以另跑递补对照，但不能改正式口径。

---

## 3. 六层诊断（全部 Development OOF）

Purge 规则与阶段3一致：用于“谁算头部”的信息，其 `label_end_time` 不得与评价样本的 `feature_available_at` 重叠。

### L1 贡献洛伦兹

按股票把 OOF 贡献排序，画三条累计曲线：

- 累计 RankIC
- 累计 Top-20 命中率
- 累计参考组合毛收益（成本单独披露，不写入标签）

报告产生 50% / 80% / 100% 累计 IC 所需的股票数。若 ≤15 只贡献超过一半 IC，判名字尾部。模板是 Bessembinder (2018) 的财富集中图，以及 Rabener (2018) 的个股贡献分布。

### L2 可迁移性网格

不要只看 Top-20。同一因子必须同时报告：

| 切片 | 通过标准（草案） |
|---|---|
| 分数五分位 Q5−Q1 | 符号与全样本一致 |
| 丢掉两端后的 Q4−Q2 | 不要求同等强度，但不得稳定为负 |
| 沪深300 内市值三分位 | 至少两个三分位同号 |
| 排名 1–20 / 21–50 / 51–100 / 其余 | 1–20 与 21–50 都不得为稳定负 |

只存在于极端组，按 Fama and French (2008)、Hou, Xue and Zhang (2020) 视为不可迁移。

### L3 行业内 vs 行业间

每天把因子拆成：

```text
f_raw(i,t) = f_industry(ind(i),t) + f_within(i,t)
```

- `f_industry`：当日该行业成员（Leave-One-Out）的因子均值
- `f_within`：个股减行业均值后的行内秩

分别对绝对收益、市场超额、行业超额三类标签算 RankIC。

判定：

- 行内 IC 与全池 IC 同号，且行业均值不是唯一来源 → 才允许当个股层
- 行业均值主导、行内 ≈0 或为负 → `INDUSTRY_PROXY`

这是 Asness, Porter and Stevens (2000) 的直接搬用，也是 Moskowitz and Grinblatt (1999) 在动量上的结论形式。

### L4 主题簇

申万一级之外，预注册五个主题（冻结后才能改名单，不能看完结果再加）：

1. 白酒
2. 新能源车 / 电池
3. 半导体设备
4. 大金融
5. 医药创新

每个主题做 Leave-One-Theme-Out。单一主题贡献过半或剔除后翻号 → `THEME_HEAD`。

依据：CSI300 相关网络是无标度的，少数节点和社群吸收大部分联动（Li and Zhao 2014；He et al. 2022）。

### L5 两种删头，禁止混用

| 规则 | 是否进入正式结论 | 做法 |
|---|---|---|
| 事前删头 | 是 | 每个信号日只按**当日已知分数**丢掉 Top-k（k∈{1,5,10,15}），再算剩余股票 IC 与组合 |
| 事后删头 | 否，只作红旗 | 即现行 `stock_drop_top5pct`：全样本 `score×标签` 选头 |

事前丢掉 Top-10 后 IC 翻号 → 不能作为个股残差因子。

### L6 组合层：现金 vs 递补

正式路径：头部不可交易 → 现金，不改写 t 日信号。

诊断对照（不进入 GO）：临时允许 21–40 名递补。若只有递补后才不亏，说明宽度不够（`CONSTRUCTION_ONLY`），不是优化器能修的。

Blitz, Huij and Martens (2011) 的残差动量说明：先去掉共同暴露，再排序，可降低动态因子暴露和极端截面集中。这是修复手段，不是诊断本身；只有诊断判为 `STOCK_RESIDUAL_OK` 的候选才值得做残差化。

---

## 4. 预注册判定

| 代码 | 条件 | 处置 |
|---|---|---|
| `STOCK_RESIDUAL_OK` | 行内 IC 与全池同号；事前丢掉 Top-10 不翻号；1–20 与 21–50 都非稳定负 | 允许作为个股层输入 |
| `INDUSTRY_PROXY` | 行业均值 IC 主导，行内 ≈0 或为负 | 迁到行业层；禁止重复计入个股 Alpha 与行业预算 |
| `THEME_HEAD` | 单一主题或单一行业贡献过半 | 不作全池因子 |
| `NAME_IDIOSYNCRATIC` | ≤10 只股票贡献 ≥50% IC，且行业分散 | 拒绝。这是选股故事 |
| `CONSTRUCTION_ONLY` | 全截面弱，只有 Top-20 PnL 好看 | 拒绝 |

未冻结本表之前，不得把任何新因子标成已验证。复杂行业对冲若不优于简单行业中性，停在行业中性。

---

## 5. 文献来源

### 5.1 删极端观测后溢价消失

- Knez, P. J., & Ready, M. J. (1997). On the robustness of size and book-to-market in cross-sectional regressions. *Journal of Finance*, 52(4), 1355–1382. https://doi.org/10.1111/j.1540-6261.1997.tb01113.x  
  每月修剪最极端 1% 后，Fama–French 规模溢价消失；16 个极端月份就能解释规模系数的负均值。

- Bessembinder, H. (2018). Do stocks outperform Treasury bills? *Journal of Financial Economics*, 129(3), 440–457. https://doi.org/10.1016/j.jfineco.2018.06.004  
  约 4% 的公司解释 1926 年以来全部净财富创造；多数个股贡献为零或为负。洛伦兹曲线的直接模板。

- Rabener, N. (2018). The impact of single stocks on factor returns. CFA Institute *Enterprising Investor*. https://rpc.cfainstitute.org/blogs/enterprising-investor/2018/the-impact-of-single-stocks-on-factor-returns  
  等权多空因子对普通个股不敏感，但对常驻多空两端的 FAANG 敏感。常驻头部 ≠ 随机个股。

### 5.2 行业代理、行内 vs 行间

- Moskowitz, T. J., & Grinblatt, M. (1999). Do industries explain momentum? *Journal of Finance*, 54(4), 1249–1290. https://doi.org/10.1111/0022-1082.00146  
  控制行业动量后，个股动量利润大幅下降；行业动量本身仍然强。这是“删掉行业头部后个股信号变弱”的经典结果。

- Asness, C. S., Porter, R. B., & Stevens, R. L. (2000). Predicting stock returns using industry-relative firm characteristics. AQR / SSRN. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=213872  
  把 size、BM、动量等拆成行业内与行业间。行内相对值通常更精确；行业间部分也独立存在。L3 的直接模板。

- Daniel, K., & Titman, S. (1997). Evidence on the characteristics of cross sectional variation in stock returns. *Journal of Finance*, 52(1), 1–33. https://doi.org/10.1111/j.1540-6261.1997.tb03806.x  
  高 BM 股票互相协方差，是因为同行同业/同地区，而不是独立的distress因子。

- Daniel, K., Mota, L., Rottke, S., & Santos, T. (2020). The cross-section of risk and return. *Review of Financial Studies* / NBER w24164. https://www.nber.org/papers/w24164  
  标准特征组合加载未定价的行业共同波动；对冲行业能提高 Sharpe，但行业不是唯一未定价风险。

- Arnott, R., Kalesnik, V., & Linnainmaa, J. (2023). Factor momentum. *Review of Financial Studies* (working paper 2018/2021). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3116974  
  行业动量可被因子动量解释。主题簇还要再问：是行业，还是更底层的共同因子。

### 5.3 可迁移性、等权与浓缩

- Fama, E. F., & French, K. R. (2008). Dissecting anomalies. *Journal of Finance*, 63(4), 1653–1678. https://doi.org/10.1111/j.1540-6261.2008.01371.x  
  按微盘/小盘/大盘检验异常是否可迁移。只存在于极端组或微盘的，不算稳健。

- Hou, K., Xue, C., & Zhang, L. (2020). Replicating anomalies. *Review of Financial Studies*, 33(5), 2019–2133. https://doi.org/10.1093/rfs/hhy131  
  NYSE 断点 + 市值加权后，大量异常失败。等权与全市场断点会让微盘/头部主导结果。

- Amenc, N., Goltz, F., Lodh, A., & Martellini, L. (2016). Diversified or concentrated factor tilts? *Journal of Portfolio Management*, 42(2), 64–76. https://doi.org/10.3905/jpm.2016.42.2.064  
  高度浓缩的因子倾斜抬收益也抬特质风险与换手。Top-20 是浓缩，不是分散因子组合。

- Blitz, D., Huij, J., & Martens, M. (2011). Residual momentum. *Journal of Empirical Finance*, 18(3), 506–521. https://doi.org/10.1016/j.jempfin.2011.01.003  
  用残差收益排序，动态因子暴露下降，极端截面集中减轻。这是修复候选，不是诊断本身。

### 5.4 中国与沪深300结构

- Liu, J., Stambaugh, R. F., & Yuan, Y. (2019). Size and value in China. *Journal of Financial Economics*, 137(3), 585–609. https://doi.org/10.1016/j.jfineco.2019.03.008  
  最小 30% 受壳价值污染；中国价值更应用 EP 而非 BM。你们已限制在点时沪深300，避开了壳，但没有避开主题龙头。A 股更常见的头部依赖是行业/主题贝塔被写成个股公式。

- Li, P., & Zhao, J. (2014). An analysis of the sectorial influence of CSI300 stocks within the directed network. *Physica A*, 396, 235–241.

- He, Y., et al. (2022). Undirected and directed network analysis of the Chinese stock market. *Computational Economics*, 60, 1155–?. CSI300 可分成金融、地产、制造等社群，网络对蓄意攻击脆弱。

### 5.5 次要但相关

- Fama, E. F. (1998). Market efficiency, long-term returns, and behavioral finance. *Journal of Financial Economics*, 49(3), 283–306. 等权放大微盘，市值加权反映可投资财富效应。
- Novy-Marx, R., & Velikov, M. (2016). A taxonomy of anomalies and their trading costs. *Review of Financial Studies*, 29(1), 104–147. 头部若来自昂贵换手，扣费后更容易翻号——与你们 Stage 4 成本拆分一致。

---

## 6. 和三层架构的衔接

| 诊断结果 | 市场层 | 行业层 | 个股层 |
|---|---|---|---|
| `STOCK_RESIDUAL_OK` | 不直接加 | 不直接加 | 允许进入公式集成 |
| `INDUSTRY_PROXY` | 不加 | 只进行业预算/约束 | 权重必须为 0 |
| `THEME_HEAD` | 可选为风险开关 | 主题预算上限 | 权重必须为 0 |
| `NAME_IDIOSYNCRATIC` / `CONSTRUCTION_ONLY` | 不加 | 不加 | 不加 |

默认仍然禁止：

```text
S = β_m M + β_i I + β_s A
```

把行业代理因子同时加进个股 Alpha 和行业预算，就是把头部依赖做两次。

---

## 7. 明确不做

- 不根据本诊断修改已冻结的 Stage 4 GO 阈值
- 不用 Holdout 挑选“看起来不依赖头部”的因子版本
- 不把行业代理重新包装成个股 Alpha
- 不把删头后的负收益写成软件故障
- 不恢复 REINFORCE
- 不向 LLM 发送原始行情或 Holdout

---

## 8. 建议的下一步（需你确认后才编码）

1. 冻结 L4 的五个主题成分名单（点时、带 `known_at`）。
2. 冻结 L2/L5 的数字门槛（Top-10、50% IC、21–50 非负）。
3. 只在 Development OOF 上实现诊断，输出 `HEAD_DEPENDENCE_REPORT`，不改建议 API。
4. 对你新增的三个因子逐个给出上表五类标签；等权集成前先看单因子标签。

未确认前，本文件保持 `DRAFT_NOT_FROZEN`。
