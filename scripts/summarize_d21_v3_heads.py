"""Produce a compact, auditable report from the independent head diagnostic."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/d21_v3_head_dependence_diag_20260914'


def read(name):
    return json.loads((OUT/name).read_text(encoding='utf-8'))


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])


def main():
    state = read('run_state.json')
    assert state['status']=='COMPLETED'
    rank = read('rank_summary.json')
    names = read('stock_names.json')
    heads = read('head_sets.json')
    contrib = pd.read_parquet(OUT/'stock_net_contribution.parquet')
    d = contrib.loc[contrib.experiment.eq('D')].sort_values('linked_net_contribution',ascending=False)
    influence = pd.read_parquet(OUT/'stock_rank_influence.parquet')
    industry = pd.read_parquet(OUT/'industry_attribution.parquet')
    random_ic = pd.read_parquet(OUT/'random_rank_controls.parquet').set_index('scenario')
    full = {r['scenario']:r for r in read('full_backtest_summary.json')}
    baseline = full['baseline']
    labels = {'old_proxy':'旧门禁代理头部','actual_D_net_pnl':'实际净收益头部','absolute_common_rank_delta_influence':'排名差影响头部'}
    matching = pd.DataFrame(read('random_matching_audit.json'))
    matching_summary = {'draws':len(matching),'draws_with_industry_fallback':int(matching.industry_fallbacks.gt(0).sum()),
                        'mean_history_gap_days':float(matching.mean_member_day_gap.mean()),'maximum_history_gap_days':int(matching.max_member_day_gap.max())}
    controls = []
    for h in heads:
        for family in ['uniform','matched']:
            rows = random_ic.loc[random_ic.index.str.startswith(h+'_'+family+'_')]
            r = [v for k,v in full.items() if k.startswith(h+'_'+family+'_')]
            controls.append({'head':h,'family':family,'rank_draws':len(rows),'target_delta_ic':float(random_ic.loc[h,'delta_ic']),
                'random_delta_ic_q05_q50_q95':np.quantile(rows.delta_ic,[.05,.5,.95]).tolist(),
                'rank_random_at_or_below_target_fraction':float((rows.delta_ic<=random_ic.loc[h,'delta_ic']).mean()),
                'full_draws':len(r),'full_random_delta_sharpe_q05_q50_q95':np.quantile([x['delta_sharpe'] for x in r],[.05,.5,.95]).tolist(),
                'full_random_delta_sharpe_at_or_below_target_count':sum(x['delta_sharpe']<=full[h]['delta_sharpe'] for x in r),
                'full_random_D_sharpe_at_or_below_target_count':sum(x['D']['sharpe']<=full[h]['D']['sharpe'] for x in r)})
    totals = {'D_mean_phase_total_return':float(d.linked_net_contribution.sum()),'D_positive_stock_contributions':float(d.loc[d.linked_net_contribution>0,'linked_net_contribution'].sum()),
              'D_negative_stock_contributions':float(d.loc[d.linked_net_contribution<0,'linked_net_contribution'].sum()),
              'D_top5_contribution':float(d.head(5).linked_net_contribution.sum()),'D_top10_contribution':float(d.head(10).linked_net_contribution.sum()),
              'D_max_observed_individual_weight':float(d.max_weight.max())}
    rows = influence.copy()
    rows['name'] = rows.code.map(names)
    rows.to_parquet(OUT/'named_stock_rank_influence.parquet',index=False)
    top_new = rows.sort_values(['absolute_new_influence','code'],ascending=[False,True]).head(10)
    top_delta = rows.head(10)
    summary = {'status':state,'reconciliation':read('reconciliation.json'),'rank':rank,'contribution_totals':totals,
               'top_net_contributors':d.head(10).to_dict('records'),'top_new_rank_influence':top_new.to_dict('records'),
               'top_delta_rank_influence':top_delta.to_dict('records'),'head_overlap':{a+' / '+b:len(set(heads[a])&set(heads[b])) for a in heads for b in heads if a<b},
               'industry_D':industry.loc[industry.experiment.eq('D')].sort_values('net_contribution',ascending=False).to_dict('records'),
               'controls':controls,'main_full_reruns':[v for k,v in full.items() if '_uniform_' not in k and '_matched_' not in k],
               'matching_audit':read('random_matching_audit.json'),'matching_summary':matching_summary,'data_checks':read('data_checks.json')}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    p = lambda x:f'{x*100:+.2f}%'
    pct = lambda x:f'{x*100:.1f}%'
    f4 = lambda x:f'{x:+.5f}'
    sections = ['# Wingman：D21-v3 头部依赖诊断',
        '仅诊断 Development OOF；不是新的 GO 检验。原 `D21_V3_FAILED`、85% 覆盖率规则、信号定义和 Holdout 锁均不变。',
        '## 1. 范围与核对',
        f'2018-04-03—2024-08-22，共 1,525 个账本交易日、{rank["stock_count"]} 只历史成员股票。头部组大小固定为 ceil(历史股票数×5%) = {rank["head_size"]} 只；不是每天 300 只的 5%。成本 ×1，5 折×5 个相位，A 是旧三信号普通 Top20，D 是新三信号缓冲 Top20。',
        '新建逐股账本与原 A/D 日账本核对通过：净收益、毛收益、净值、现金、持股数、换手、费用逐项误差均小于 1e-10。逐股贡献加总也通过核对。原文件未覆盖。',
        '## 2. 公平样本：先排除缺失率干扰',
        f'共同有效样本 {rank["common_rows"]:,} 个股票日，占这 1,525 个账本日期的原 OOF 成员日 {rank["oof_member_rows"]:,} 的 {rank["common_rows"]/rank["oof_member_rows"]:.1%}。RankIC 实际有 1,520 个有效日期（末端 5 日标签不足）。这里只缩小比较样本，不改变原门禁完整 OOF 的 460,500 成员日分母，也不修改交易规则。',
        table(['信号','原各自有效样本 RankIC','共同有效样本 RankIC'],[[s,f4(rank['own_sample_ic'][s]),f4(rank['common_sample_ic'][s])] for s in ['simple_ensemble','slow_residual_ensemble','IDIO_LOW_VOL_60','RES_MOM_120_20','RES_TREND_EFF_60_5']]),
        '旧三信号 = 20 日动量 MOM_20、5 日反转 REV_5、20 日低波 LOW_VOL_20；分别截面缩尾、百分位排序，三项完整才等权平均。共同样本不等于重新训练或重做信号。',
        f'保留旧头部剔除：原各自样本新−旧 RankIC 差从 {f4(rank["old_test"][0]["delta_ic"])} 到 {f4(rank["old_test"][1]["delta_ic"])}；换成公平共同样本后，从 {f4(float(random_ic.loc["baseline","delta_ic"]))} 到 {f4(float(random_ic.loc["old_proxy","delta_ic"]))}。因此原来的翻号不能单纯归咎于新旧缺失样本不同。单项信号仍只有低波明显强于旧集成；这是诊断线索，不是删除其他因子的充分依据。',
        '## 3. 实际交易净收益头部',
        '以下为回测中真实持有、按原成本扣费后的逐股净贡献。单位为整个 OOF 期间的期末净值百分点，五个相位取平均；不是年化收益贡献，也不是实际实盘利润。',
        table(['股票','名称','净贡献（百分点）','全期平均权重','最高观察权重'],[[r.code,r['name'],f'{r.linked_net_contribution*100:+.2f}',pct(r.mean_weight_all_phase_days),pct(r.max_weight)] for _,r in d.head(10).iterrows()]),
        f'全部股票净贡献合计 {p(totals["D_mean_phase_total_return"])}；正贡献合计 {p(totals["D_positive_stock_contributions"])}，负贡献合计 {p(totals["D_negative_stock_contributions"])}。前 5 / 前 10 只合计分别为 {p(totals["D_top5_contribution"])} / {p(totals["D_top10_contribution"])}。因存在亏损股票，赢家占净收益比例可能超过 100%，不能当作仓位占比。',
        '## 4. 排名影响头部',
        '逐股移除后，在剩余共同样本内重新计算带并列平均秩的 Spearman 相关。表中变化 = 移除后 − 移除前；负数表示移除该股令指标下降。按绝对变化排序，可同时看到帮助或拖累排名的股票。',
        '### 对新信号自身 RankIC 影响最大的股票',
        table(['股票','名称','新 RankIC 变化','新−旧 RankIC 差的变化'],[[r.code,r['name'],f4(r.new_ic_change),f4(r.delta_ic_change)] for _,r in top_new.iterrows()]),
        '### 对新−旧 RankIC 差影响最大的股票',
        table(['股票','名称','新−旧 RankIC 差的变化'],[[r.code,r['name'],f4(r.delta_ic_change)] for _,r in top_delta.iterrows()]),
        f'全部逐股剔除后的新−旧 RankIC 差范围：{f4(rank["common_loo_min_delta"])} 至 {f4(rank["common_loo_max_delta"])}。这用于区分单只股票和一组股票效应；仅三个头部组的首位股票做了额外完整交易回测，不应声称每只股票都做了完整回测。',
        '## 5. 保留原测试 + 随机对照 + 完整回测',
        '旧头部按每股 mean(新信号分数×5 日收益标签) 选出，属于事后代理排名，不是真实交易贡献。完整回测固定原信号，移除交易候选，重新排序、建仓、调仓并执行现金、成本、停牌、涨跌停及成员退出约束；不直接从原收益中扣去赢家。残差参考池不改变。',
        table(['剔除情景','D 净 Sharpe','D 扣费年化','D−A Sharpe差','D−A 年化差（百分点）'],[[('未剔除' if k=='baseline' else labels.get(k,k)),f'{v["D"]["sharpe"]:.3f}',p(v['D']['annualized_return']),f'{v["delta_sharpe"]:+.3f}',f'{v["delta_annualized"]*100:+.2f}'] for k,v in full.items() if '_uniform_' not in k and '_matched_' not in k]),
        '每个头部组各 200 次同数量均匀随机剔除 + 200 次行业/历史长度匹配剔除。每类前 10 次固定做完整回测，没有挑选有利的随机结果。匹配使用 OOF 主行业，历史长度从最接近的 10 只中抽取；不足时行业回退，详情见 matching_audit。匹配对照排除目标头部，均匀对照允许抽到部分头部，二者回答不同问题。',
        f'匹配质量：{matching_summary["draws"]} 次匹配抽样中，{matching_summary["draws_with_industry_fallback"]} 次发生至少一次行业回退；平均历史长度绝对差 {matching_summary["mean_history_gap_days"]:.1f} 个成员日，最大差 {matching_summary["maximum_history_gap_days"]} 日。若历史支持不足，不能把它称为严格同条件对照。',
        table(['头部组','随机对照','定向剔除后共同 ΔIC','随机 ΔIC 5% / 50% / 95%','随机 ΔIC≤定向结果','完整回测随机 ΔSharpe≤定向结果'],[[labels[r['head']],('均匀' if r['family']=='uniform' else '行业/长度匹配'),f4(r['target_delta_ic']),' / '.join(f4(v) for v in r['random_delta_ic_q05_q50_q95']),pct(r['rank_random_at_or_below_target_fraction']),str(r['full_random_delta_sharpe_at_or_below_target_count'])+'/10'] for r in controls]),
        '这些比例是描述性压力诊断，不是显著性 p 值。头部由同一段 Development 的结果事后选出，删赢家本来就可能变差；10 次完整随机回测不足以精确推断分布尾部。',
        '## 6. 行业与数据核查',
        table(['行业','净贡献（百分点）','全期平均持仓权重'],[[r.industry_name,f'{r.net_contribution*100:+.2f}',pct(r.mean_weight)] for _,r in industry.loc[industry.experiment.eq('D')].sort_values('net_contribution',ascending=False).head(10).iterrows()]),
        f'已查特征、行情、状态重复行；发现 {summary["data_checks"]["flag_count"]} 条头部股票价格异常候选（大幅波动/原价与复权价分歧），保存在 price_flags_not_confirmed_errors.parquet。它们可能是正常公司行动、时间间隔或真实波动，尚不能判定数据错误，也未修改或删除。',
        '5% 是调仓目标权重，不是每天硬上限：价格漂移、交易限制会导致观察权重变化。行业贡献集中也不等于行业风险解释了全部 Alpha；本报告未做风险因子回归，不能直接认定应行业中性化。',
        '## 7. 方法边界与下一步',
        '- 数据错误：对异常候选逐项核对来源和公司行动后再决定是否修订，保留修订记录。不能因表现极端就删除。\n- 行业效应：先做行业内与行业间收益分解，再预注册行业权重约束实验，不默认完全中性。\n- 仓位风险：以实际持仓和风险贡献评估，不将目标 5% 当成每日绝对上限。\n- 事后赢家：需要未参与选法的后续数据或前向影子运行补证；本次不打开 Holdout。\n- 弱信号：共同样本 RankIC 仍不能替代固定组合消融；未据此删动量/趋势，也未搜索新公式。',
        '所有剔除情景都是反事实压力测试，并非可事前执行的删股清单。完整回测遵从原 OOF 各折独立现金起点和相位收益拼接；不是不中断实盘账户。新诊断不撤销原失败，不重设经济/稳健性门槛，不进入 Stage 5。',
        '## 可复核产物',
        '`diagnostic_plan.json` 固定方法；`input_hashes.json` 输入/代码指纹；`reconciliation.json` 账本核对；`summary.json` 全部结构化摘要；`baseline_stock_day_attribution.parquet` 逐股日收益/成本/仓位；`stock_rank_influence.parquet` 全部逐股影响；`random_deletion_sets.json` 全部固定抽样；`full_backtest_summary.json` 完整回测指标；`backtest_*_daily.parquet` 各情景日账本。'
    ]
    (OUT/'REPORT.md').write_text('\n\n'.join(sections)+'\n',encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['contribution_totals','controls','head_overlap','main_full_reruns']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
