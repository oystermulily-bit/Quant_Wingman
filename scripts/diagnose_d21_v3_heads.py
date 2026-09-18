"""Independent, read-only-to-inputs Development diagnostic. See fixed diagnostic_plan.json.

Run with python -B scripts/diagnose_d21_v3_heads.py. No production modules modified.
Original run methods are instrumented in memory, preserving all trading operations.
Cached price lookup and target sorting are checked against the frozen daily ledger.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
import textwrap
import time
from collections import defaultdict
from pathlib import Path
from types import MethodType

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from research_stage3.backtest import ReferenceBacktester
from research_d21_v2.backtest import BufferedTop20Backtester
from research_d21_v3_execution import D21V3Config
from research_stage4.runner import _rank_ic
from research_stage4.stats import performance_from_returns

OUT = ROOT / 'experiments/d21_v3_head_dependence_diag_20260914'
SRC = ROOT / 'experiments/stage3r_d21_v3_reference_only_h5'
SNAP = Path('D:/Hulucoding/AmAzing_Data/research_snapshots/csi300_2014_present_v2')
OLD = 'simple_ensemble'
NEW = 'slow_residual_ensemble'
COMP = ['IDIO_LOW_VOL_60', 'RES_MOM_120_20', 'RES_TREND_EFF_60_5']
CFG = D21V3Config()


def emit(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def filtered_ranks(values, keep):
    """Exact average-tie ranks of each surviving row, with removed positions zero."""
    order = np.argsort(values, kind='stable')
    ordered = values[order]
    lo = np.searchsorted(ordered, ordered, side='left')
    hi = np.searchsorted(ordered, ordered, side='right')
    mask = keep[:, order]
    prefix = np.pad(np.cumsum(mask, axis=1), ((0, 0), (1, 0)))
    ranks = ((prefix[:, lo] + prefix[:, hi] + 1) * .5) * mask
    return ranks[:, np.argsort(order)]


def correlations(x, y, keep):
    rx, ry = filtered_ranks(x, keep), filtered_ranks(y, keep)
    n = keep.sum(axis=1)
    den = np.maximum(n, 1)
    sx, sy = rx.sum(axis=1), ry.sum(axis=1)
    cov = (rx * ry).sum(axis=1) - sx * sy / den
    vx = (rx * rx).sum(axis=1) - sx * sx / den
    vy = (ry * ry).sum(axis=1) - sy * sy / den
    out = np.full(len(keep), np.nan)
    good = (n >= 10) & (vx > 1e-12) & (vy > 1e-12)
    out[good] = cov[good] / np.sqrt(vx[good] * vy[good])
    return out


def rank_diagnostics(frame, scenarios, common=True):
    codes = sorted(frame.code.unique())
    code_idx = {v: i for i, v in enumerate(codes)}
    drop = np.zeros((len(scenarios), len(codes)), dtype=bool)
    for i, deleted in enumerate(scenarios.values()):
        drop[i, [code_idx[c] for c in deleted if c in code_idx]] = True
    sums = np.zeros((len(scenarios), 2))
    counts = np.zeros_like(sums)
    rows = np.zeros_like(sums)
    for _, block in frame.groupby('signal_date', sort=True):
        data = block[[OLD, NEW, 'value']].to_numpy(dtype=float)
        ids = np.array([code_idx[c] for c in block.code])
        for j in range(2):
            valid = np.isfinite(data).all(axis=1) if common else np.isfinite(data[:, [j, 2]]).all(axis=1)
            if valid.sum() < 10:
                continue
            mask = ~drop[:, ids[valid]]
            corrs = correlations(data[valid, j], data[valid, 2], mask)
            sums[:, j] += np.nan_to_num(corrs)
            counts[:, j] += np.isfinite(corrs)
            rows[:, j] += mask.sum(axis=1)
    means = sums / np.maximum(counts, 1)
    result = pd.DataFrame({'scenario': list(scenarios), 'old_ic': means[:, 0], 'new_ic': means[:, 1],
                          'delta_ic': means[:, 1] - means[:, 0], 'old_dates': counts[:, 0],
                          'new_dates': counts[:, 1], 'old_rows': rows[:, 0], 'new_rows': rows[:, 1]})
    return result


def selftest_ranks():
    rng = np.random.default_rng(19)
    for n in [9, 10, 11, 39]:
        x, y = rng.integers(0, 7, (2, n)).astype(float)
        masks = rng.random((12, n)) > .15
        got = correlations(x, y, masks)
        for m, actual in zip(masks, got):
            expected = pd.Series(x[m]).rank().corr(pd.Series(y[m]).rank()) if sum(m) >= 10 else np.nan
            assert np.allclose(actual, expected, atol=1e-12, equal_nan=True), (actual, expected)
    emit('Exact average-tie deletion-rank tests passed')


def instrument_run(cls):
    source = textwrap.dedent(inspect.getsource(cls.run))
    # Insert observations, not trading decisions. Assert each insertion count.
    assert source.count('day_cost += fee') == 2
    assert source.count('gross_day_gain += gain') == 1
    lines = []
    for line in source.splitlines():
        lines.append(line)
        indent = line[:len(line) - len(line.lstrip())]
        if line.strip() == 'day_cost += fee':
            lines.append(indent + 'self._observe(date, code, 0.0, fee, previous_nav, 0.0, None)')
        if line.strip() == 'gross_day_gain += gain':
            lines.append(indent + 'self._observe(date, code, gain, 0.0, previous_nav, value, asset_return)')
    namespace = dict(cls.run.__globals__)
    exec(compile('\n'.join(lines), '<diagnostic-only instrumented '+cls.__name__+'>', 'exec'), namespace)
    return namespace['run']


class Attribution:
    def __init__(self):
        self.rows = {}
        self.enabled = False

    def observe(self, date, code, gain, fee, nav, value, ret):
        if not self.enabled:
            return
        key = (date, code)
        row = self.rows.setdefault(key, [0., 0., 0., 0., np.nan])
        row[0] += gain / nav
        row[1] += fee / nav
        row[2] += value / nav
        row[3] += fee
        if ret is not None:
            row[4] = ret


class Books:
    def __init__(self, features, bars, status, dates, folds):
        self.f, self.b, self.s, self.dates, self.folds = features, bars, status, dates, folds
        self.membership = features[['date', 'code']]
        self.naive = ReferenceBacktester(CFG.as_stage3_config())
        self.buffer = BufferedTop20Backtester(CFG)
        self.naive._prepare_market(features, bars, status, dates, self.membership)
        self.buffer._naive = self.naive
        # Cache only representations, not stateful decisions or holdings.
        px = self.naive._open_prices
        matrix, di, ci = px.to_numpy(), {d: i for i, d in enumerate(px.index)}, {c: i for i, c in enumerate(px.columns)}
        def lookup(_self, date, code):
            r, c = di.get(date), ci.get(code)
            return float(matrix[r, c]) if r is not None and c is not None else float('nan')
        self.naive._open_px = MethodType(lookup, self.naive)
        self.naive._prepare_market = lambda *a, **k: self.dates
        self.raw_orders = {OLD: {}, NEW: {}}
        self.liquid = {}
        for date, block in features.groupby('date', sort=True):
            self.liquid[date] = set(block.loc[block.median_amount_20.ge(CFG.liquidity_median_amount_20d), 'code'])
            for score, valid in [(OLD, 'valid_signal'), (NEW, 'valid_slow_signal')]:
                eligible = block.loc[block[valid].fillna(False).astype(bool)].sort_values([score, 'code'], ascending=[False, True], kind='stable')
                self.raw_orders[score][date] = list(zip(eligible.code, eligible[score]))
        self.naive._prepare_targets = self.targets
        self.buffer._buffered_targets = self.buffer_targets
        self.attr = Attribution()
        for obj, cls in [(self.naive, ReferenceBacktester), (self.buffer, BufferedTop20Backtester)]:
            obj.run = MethodType(instrument_run(cls), obj)
            obj._observe = self.attr.observe

    def set_deletion(self, deleted):
        self.orders = {s: {d: [(c, v) for c, v in a if c not in deleted] for d, a in bydate.items()} for s, bydate in self.raw_orders.items()}
        self.ranks = {d: {c: i + 1 for i, (c, _) in enumerate(a)} for d, a in self.orders[NEW].items()}

    def targets(self, dates, allowed_signals, *, horizon, phase, score_column):
        result = {}
        for i, d in enumerate(dates[:-1]):
            if d not in allowed_signals or i % horizon != phase:
                continue
            result[dates[i+1]] = [(c, s) for c, s in self.orders[score_column].get(d, []) if c in self.liquid[d]][:CFG.top_n]
        return result

    def buffer_targets(self, *, signal_date, universe, holdings, nav_before, score_column, valid_column):
        ranks = self.ranks.get(signal_date, {})
        held = [c for c, v in holdings.items() if v > 1e-12]
        targets = [c for c in held if c in universe and ranks.get(c, 10**9) <= CFG.exit_rank]
        if len(held) <= CFG.top_n:
            for c, _ in self.orders[score_column].get(signal_date, [])[:CFG.top_n]:
                if len(targets) >= CFG.top_n:
                    break
                if c not in targets and c in universe and c in self.liquid[signal_date]:
                    targets.append(c)
        scale = nav_before if nav_before > 0 else 1.
        assert len(targets) * CFG.target_weight <= 1 + 1e-12
        return {c: scale * CFG.target_weight for c in targets}, ranks

    def run(self, name, deleted, attribution=False):
        self.set_deletion(deleted)
        outputs, events, metrics = [], [], []
        self.attr.enabled = attribution
        for exp, book, score in [('A', self.naive, OLD), ('D', self.buffer, NEW)]:
            for fold in self.folds:
                signals = self.dates[(self.dates >= fold['validation_start']) & (self.dates <= fold['validation_end'])]
                for phase in range(5):
                    self.attr.rows = {}
                    m, daily = book.run(self.f, self.b, self.s, self.dates, signals, horizon=5, phase=phase, cost_multiplier=1., score_column=score, membership=self.membership)
                    daily['experiment'], daily['fold_id'], daily['phase'] = exp, fold['fold_id'], phase
                    outputs.append(daily)
                    metrics.append(dict(m.to_dict(), experiment=exp, fold_id=fold['fold_id'], scenario=name))
                    if attribution:
                        for (date, code), v in self.attr.rows.items():
                            events.append((date, code, exp, fold['fold_id'], phase, *v))
        daily = pd.concat(outputs, ignore_index=True)
        summary = {}
        for exp, subset in daily.groupby('experiment'):
            phase_rows = []
            for phase, block in subset.groupby('phase'):
                block = block.sort_values('date')
                phase_rows.append(dict(phase=int(phase), **performance_from_returns(block.net_return, annual_days=239)))
            summary[exp] = {'phases': phase_rows, **{k: float(np.median([p[k] for p in phase_rows])) for k in ['sharpe', 'annualized_return', 'max_drawdown', 'total_return']}}
        summary['delta_sharpe'] = summary['D']['sharpe'] - summary['A']['sharpe']
        summary['delta_annualized'] = summary['D']['annualized_return'] - summary['A']['annualized_return']
        summary['scenario'] = name
        summary['deleted'] = sorted(deleted)
        event_frame = pd.DataFrame(events, columns=['date', 'code', 'experiment', 'fold_id', 'phase', 'gain_return', 'fee_return', 'holding_weight', 'fee_cash', 'asset_return'])
        return summary, daily, event_frame, metrics


def random_sets(heads, meta, plan):
    rng = np.random.default_rng(plan['seed'])
    universe = sorted(meta.index)
    sets, audits = {}, []
    for head_name, head in heads.items():
        for i in range(plan['random_controls_per_head']):
            sets[f'{head_name}_uniform_{i:03}'] = set(rng.choice(universe, len(head), replace=False))
            available, selected = set(universe) - head, set()
            fallbacks, gaps = 0, []
            for target in sorted(head):
                pool = [c for c in sorted(available) if meta.loc[c, 'industry_code'] == meta.loc[target, 'industry_code']]
                if not pool:
                    pool, fallbacks = sorted(available), fallbacks + 1
                near = sorted(pool, key=lambda c: (abs(meta.loc[c, 'member_days']-meta.loc[target, 'member_days']), c))[:10]
                chosen = str(rng.choice(near))
                gaps.append(abs(int(meta.loc[chosen, 'member_days']) - int(meta.loc[target, 'member_days'])))
                available.remove(chosen)
                selected.add(chosen)
            key = f'{head_name}_matched_{i:03}'
            sets[key] = selected
            audits.append({'scenario': key, 'industry_fallbacks': fallbacks, 'mean_member_day_gap': float(np.mean(gaps)), 'max_member_day_gap': max(gaps)})
    return sets, audits


def main():
    plan = json.loads((OUT/'diagnostic_plan.json').read_text(encoding='utf-8'))
    save('run_state.json', {'status': 'RUNNING', 'holdout_read': False, 'stage5_allowed': False})
    selftest_ranks()
    frozen = json.loads((SRC/'run_plan.json').read_text(encoding='utf-8'))
    for p, sha in frozen['implementation_hashes'].items():
        assert digest(ROOT/p) == sha, f'Frozen implementation changed: {p}'
    split = json.loads((SRC/'split_plan.json').read_text(encoding='utf-8'))
    cutoff = pd.Timestamp(split['holdout_start'])
    source_paths = [SRC/x for x in ['development_features.parquet','development_labels.parquet','development_oof_portfolio_daily.parquet','split_plan.json','run_plan.json']]
    source_paths += [SNAP/'standardized'/x for x in ['execution_bars.parquet','execution_status.parquet','security_master.parquet']]
    save('input_hashes.json', {str(p): digest(p) for p in source_paths} | {'diagnostic_plan': digest(OUT/'diagnostic_plan.json'), 'script': digest(Path(__file__))})
    cols = ['date','code','industry_code','industry_name','median_amount_20',OLD,NEW,'valid_signal','valid_slow_signal', *COMP]
    f = pd.read_parquet(SRC/'development_features.parquet', columns=cols)
    assert f.date.max() < cutoff and not f.duplicated(['date','code']).any()
    labels = pd.read_parquet(SRC/'development_labels.parquet', columns=['signal_date','code','value','label_type','horizon'])
    assert labels.signal_date.max() < cutoff
    labels = labels.loc[labels.label_type.eq('ABSOLUTE') & labels.horizon.eq(5)]
    daily = pd.read_parquet(SRC/'development_oof_portfolio_daily.parquet')
    assert daily.date.max() < cutoff
    oof_dates = pd.DatetimeIndex(sorted(daily.date.unique()))
    of = f.loc[f.date.isin(oof_dates)].copy()
    joined = of.merge(labels, left_on=['date','code'], right_on=['signal_date','code'], validate='one_to_one')
    assert joined.signal_date.nunique() == 1525
    common = joined.replace([np.inf,-np.inf],np.nan).dropna(subset=[OLD,NEW,'value'])
    emit(f'OOF rows={len(joined)}, common rows={len(common)}')
    own_ics = {s: _rank_ic(joined, s) for s in [OLD, NEW, *COMP]}
    common_ics = {s: _rank_ic(common, s) for s in [OLD, NEW, *COMP]}
    pairwise = {}
    for s in COMP:
        pair = joined.replace([np.inf,-np.inf],np.nan).dropna(subset=[OLD,s,'value'])
        pairwise[s] = {'old_ic': _rank_ic(pair,OLD), 'new_ic': _rank_ic(pair,s), 'rows': len(pair)}
    allcodes = sorted(joined.code.unique())
    n_head = max(1,int(np.ceil(.05*len(allcodes))))
    proxy = joined.assign(proxy=joined[NEW]*pd.to_numeric(joined.value, errors='coerce')).groupby('code').proxy.mean().sort_values(ascending=False)
    old_head = set(proxy.head(n_head).index)
    proxy.rename('old_proxy').reset_index().to_parquet(OUT/'old_proxy_stock_order.parquet',index=False)
    scenarios = {'baseline':set(), 'old_proxy':old_head, **{f'loo:{c}': {c} for c in allcodes}}
    emit(f'Computing exact leave-one-out RankIC for {len(allcodes)} stocks')
    loo = rank_diagnostics(joined, scenarios, common=True)
    loo.to_parquet(OUT/'common_leave_one_out.parquet',index=False)
    base = loo.iloc[0]
    assert abs(base.old_ic-common_ics[OLD])<1e-12 and abs(base.new_ic-common_ics[NEW])<1e-12
    influence = loo.loc[loo.scenario.str.startswith('loo:')].copy()
    influence['code'] = influence.scenario.str[4:]
    influence['new_ic_change'] = influence.new_ic-base.new_ic
    influence['delta_ic_change'] = influence.delta_ic-base.delta_ic
    influence['absolute_delta_influence'] = abs(influence.delta_ic_change)
    influence['absolute_new_influence'] = abs(influence.new_ic_change)
    influence = influence.sort_values(['absolute_delta_influence','code'],ascending=[False,True])
    influence.to_parquet(OUT/'stock_rank_influence.parquet',index=False)
    old_own = rank_diagnostics(joined, {'baseline':set(),'old_proxy':old_head}, common=False)
    old_own.to_parquet(OUT/'original_separate_mask_test.parquet',index=False)
    save('rank_summary.json', {'own_sample_ic':own_ics,'common_sample_ic':common_ics,'pairwise_component_ic':pairwise,'oof_member_rows':len(of),'joined_rows':len(joined),'common_rows':len(common),'stock_count':len(allcodes),'head_size':n_head,'common_loo_min_delta':float(influence.delta_ic.min()),'common_loo_max_delta':float(influence.delta_ic.max()),'old_test':old_own.to_dict('records')})
    # Explicit Development predicates: never open sealed or Holdout outcome files.
    bars = pd.read_parquet(SNAP/'standardized/execution_bars.parquet', columns=['date','code','open_tr','open_raw','has_quote'], filters=[('date','<',cutoff)])
    status = pd.read_parquet(SNAP/'standardized/execution_status.parquet', columns=['date','code','can_buy_open','can_sell_open'], filters=[('date','<',cutoff)])
    assert bars.date.max()<cutoff and status.date.max()<cutoff
    assert not bars.duplicated(['date','code']).any() and not status.duplicated(['date','code']).any()
    names = pd.read_parquet(SNAP/'standardized/security_master.parquet', columns=['code','name']).drop_duplicates('code').set_index('code').name.to_dict()
    save('stock_names.json', names)
    dates = pd.DatetimeIndex(sorted(f.date.unique()))
    assert len(dates)==split['development_date_count']
    emit('Preparing unchanged execution rules and baseline per-stock accounting')
    books = Books(f,bars,status,dates,split['folds'])
    bs, bd, ev, bm = books.run('baseline',set(),attribution=True)
    keys = ['date','experiment','fold_id','phase']
    expected = daily.loc[daily.experiment.isin(['A','D'])].sort_values(keys).set_index(keys)
    actual = bd.sort_values(keys).set_index(keys)
    assert expected.index.equals(actual.index)
    checks = {}
    for col in ['net_return','gross_return','nav','cash_weight','holding_count','turnover','transaction_cost']:
        error = float(np.max(np.abs(expected[col].to_numpy()-actual[col].to_numpy())))
        checks[col] = error
        assert error < 1e-10,(col,error)
    ev['net_contribution_daily'] = ev.gain_return-ev.fee_return
    recon = ev.groupby(keys).net_contribution_daily.sum().reindex(actual.index,fill_value=0)
    checks['stock_daily_contribution_sum'] = float(np.max(abs(recon.to_numpy()-actual.net_return.to_numpy())))
    assert checks['stock_daily_contribution_sum'] < 1e-10
    bd = bd.sort_values(['experiment','phase','date'])
    bd['link_nav'] = bd.groupby(['experiment','phase']).net_return.transform(lambda x:(1+x).cumprod().shift(fill_value=1.))
    ev = ev.merge(bd[keys+['link_nav']],on=keys,validate='many_to_one')
    ev['linked_net_contribution'] = ev.net_contribution_daily*ev.link_nav/5.
    ev['linked_gross_contribution'] = ev.gain_return*ev.link_nav/5.
    ev['linked_cost_contribution'] = ev.fee_return*ev.link_nav/5.
    ev = ev.merge(f[['date','code','industry_code','industry_name']],on=['date','code'],how='left',validate='many_to_one')
    # Exited positions can persist under sell constraints; carry last *past* industry only.
    industry_history = f[['date','code','industry_code','industry_name']].sort_values('date')
    missing = ev.industry_code.isna()
    if missing.any():
        repaired = pd.merge_asof(ev.loc[missing,['date','code']].sort_values('date'),industry_history,on='date',by='code',direction='backward')
        repaired = repaired.set_index(['date','code'])
        for col in ['industry_code','industry_name']:
            ev.loc[missing,col] = [repaired.loc[(d,c),col].iloc[0] if isinstance(repaired.loc[(d,c),col],pd.Series) else repaired.loc[(d,c),col] for d,c in ev.loc[missing,['date','code']].itertuples(index=False,name=None)]
    ev.to_parquet(OUT/'baseline_stock_day_attribution.parquet',index=False)
    bd.to_parquet(OUT/'baseline_reproduced_daily.parquet',index=False)
    save('baseline_book_metrics.json',bm)
    totals = ev.groupby(['experiment','code'])[['linked_net_contribution','linked_gross_contribution','linked_cost_contribution']].sum().reset_index()
    exposure = ev.groupby(['experiment','code']).agg(weight_sum=('holding_weight','sum'),max_weight=('holding_weight','max'),held_phase_days=('holding_weight',lambda x:int((x>0).sum())))
    totals = totals.merge(exposure,on=['experiment','code'])
    totals['mean_weight_all_phase_days'] = totals.weight_sum/(len(oof_dates)*5)
    totals['name'] = totals.code.map(names)
    totals = totals.sort_values(['experiment','linked_net_contribution','code'],ascending=[True,False,True])
    totals.to_parquet(OUT/'stock_net_contribution.parquet',index=False)
    for exp in ['A','D']:
        total = totals.loc[totals.experiment.eq(exp),'linked_net_contribution'].sum()
        expected_total = np.mean([p['total_return'] for p in bs[exp]['phases']])
        checks[f'{exp}_linked_total_error'] = float(abs(total-expected_total))
        assert checks[f'{exp}_linked_total_error']<1e-10
    save('reconciliation.json',{'passed':True,'tolerance':1e-10,'errors':checks,'original_reports_unchanged':True,'holdout_read':False})
    emit(f'Baseline accounting reconciled; maximum error={max(checks.values()):.3g}')
    industry = ev.groupby(['experiment','industry_code','industry_name'],dropna=False).agg(net_contribution=('linked_net_contribution','sum'),gross_contribution=('linked_gross_contribution','sum'),cost_contribution=('linked_cost_contribution','sum'),weight_sum=('holding_weight','sum')).reset_index()
    industry['mean_weight'] = industry.weight_sum/(len(oof_dates)*5)
    industry.to_parquet(OUT/'industry_attribution.parquet',index=False)
    meta = of.groupby('code').agg(member_days=('date','nunique'),industry_code=('industry_code',lambda x: x.dropna().mode().iloc[0] if len(x.dropna()) else 'UNKNOWN'))
    meta.to_parquet(OUT/'stock_matching_metadata.parquet')
    head_actual = set(totals.loc[totals.experiment.eq('D')].head(n_head).code)
    assert len(head_actual)==n_head
    heads = {'old_proxy':old_head,'actual_D_net_pnl':head_actual,'absolute_common_rank_delta_influence':set(influence.head(n_head).code)}
    save('head_sets.json',{k:sorted(v) for k,v in heads.items()})
    rnd, audits = random_sets(heads,meta,plan)
    save('random_deletion_sets.json',{k:sorted(v) for k,v in rnd.items()})
    save('random_matching_audit.json',audits)
    emit('Computing 1,200 fixed random-removal common-sample RankIC controls')
    ri = rank_diagnostics(joined,{'baseline':set(),**heads,**rnd},common=True)
    ri.to_parquet(OUT/'random_rank_controls.parquet',index=False)
    # Flag, do not repair, adjustment/price anomalies at head securities actually held.
    px = bars.sort_values(['code','date']).copy()
    px['adjusted_return'] = px.groupby('code').open_tr.pct_change(fill_method=None)
    px['raw_return'] = px.groupby('code').open_raw.pct_change(fill_method=None)
    px['adjustment_divergence'] = px.adjusted_return-px.raw_return
    flags = px.loc[px.code.isin(set.union(*heads.values())) & ((px.adjusted_return.abs()>.25)|(px.adjustment_divergence.abs()>.10)|(px.open_tr<=0))].copy()
    flags.to_parquet(OUT/'price_flags_not_confirmed_errors.parquet',index=False)
    ev.loc[ev.experiment.eq('D') & ev.code.isin(set.union(*heads.values()))].assign(abs_return=lambda x:x.asset_return.abs()).sort_values('abs_return',ascending=False).head(100).to_parquet(OUT/'largest_held_head_return_intervals.parquet',index=False)
    save('data_checks.json',{'duplicate_feature_rows':0,'duplicate_execution_bar_rows':0,'duplicate_status_rows':0,'flag_count':len(flags),'flags_are_not_proof_of_error':True,'label_nonfinite_rows':int((~np.isfinite(joined.value)).sum()),'source_correction_performed':False})
    leaders = {'old_proxy':str(proxy.index[0]),'actual_D_net_pnl':str(totals.loc[totals.experiment.eq('D')].iloc[0].code),'absolute_common_rank_delta_influence':str(influence.iloc[0].code)}
    fullsets = {**heads,**{f'single_leader_{k}':{v} for k,v in leaders.items()},**{k:v for k,v in rnd.items() if int(k.rsplit('_',1)[1])<plan['random_full_backtests_per_head']}}
    save('full_rerun_deletion_sets.json',{k:sorted(v) for k,v in fullsets.items()})
    results = [bs]
    save('full_backtest_summary.json',results)
    for i,(name,removed) in enumerate(fullsets.items()):
        checkpoint = OUT/f'backtest_{name}.json'
        if checkpoint.exists():
            r = json.loads(checkpoint.read_text(encoding='utf-8'))
            assert r['deleted']==sorted(removed)
        else:
            r, d, _, metrics = books.run(name,removed)
            d.to_parquet(OUT/f'backtest_{name}_daily.parquet',index=False)
            save(f'backtest_{name}_book_metrics.json',metrics)
            save(checkpoint.name,r)
        results.append(r)
        save('full_backtest_summary.json',results)
        save('run_state.json',{'status':'RUNNING_FULL_RERUNS','completed':i+1,'total':len(fullsets),'holdout_read':False,'stage5_allowed':False})
        emit(f'Full rerun {i+1}/{len(fullsets)} {name}: D Sharpe={r["D"]["sharpe"]:.4f}, D-A={r["delta_sharpe"]:.4f}')
    # Detect input edits made by other tasks during this long-running diagnostic.
    hashes = json.loads((OUT/'input_hashes.json').read_text(encoding='utf-8'))
    for p in source_paths:
        assert digest(p)==hashes[str(p)],f'Input changed during diagnosis: {p}'
    for p, sha in frozen['implementation_hashes'].items():
        assert digest(ROOT/p)==sha,f'Frozen implementation changed during diagnosis: {p}'
    save('run_state.json',{'status':'COMPLETED','full_reruns':len(fullsets),'holdout_read':False,'stage5_allowed':False,'gate_unchanged':'D21_V3_FAILED'})
    emit('All diagnostics complete; original gate and artifacts unchanged')


if __name__=='__main__':
    main()
