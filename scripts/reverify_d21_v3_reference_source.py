"""Fresh Development-only source verification with real request lineage; no secrets in artifacts."""
import contextlib
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_d21_v3_preflight import fingerprint
from research_d21_v3_tech_expansion import expand_intervals, additions, normalize_returns

OLD = ROOT / 'experiments/d21_v3_csi500_tech_preflight_20260914'
OUT = ROOT / 'experiments/d21_v3_source_reverified_20260914'


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def flatten(value):
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if not isinstance(value, dict):
        raise ValueError('Expected dictionary/DataFrame')
    return pd.concat([f.assign(_response_key=str(k)) for k, f in value.items()
                      if isinstance(f, pd.DataFrame)], ignore_index=True)


def main():
    OUT.mkdir(exist_ok=False)
    ledger, client = [], None
    report = {'status': 'REVERIFY_STARTED', 'holdout_read': False, 'labels_read': False,
              'economic_gate_evaluated': False, 'started_at': utcnow()}
    write_json(OUT / 'report.json', report)
    try:
        config = json.loads(Path('C:/Users/Administrator/.cursor/mcp.json').read_text(encoding='utf-8'))
        for key in ['AD_USERNAME', 'AD_PASSWORD', 'AD_HOST', 'AD_PORT']:
            os.environ[key] = str(config['mcpServers']['amazingdata']['env'][key])
        os.environ['NUMBA_DISABLE_JIT'] = '1'
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from data_pipeline.hs300.sdk_client import AmazingDataClient
            client = AmazingDataClient()
            client.login()
        print('SOURCE_LOGIN_COMPLETE', flush=True)

        def request(name, method, codes, **kwargs):
            row = {'request_id': name, 'snapshot_id': OUT.name, 'method': method.__name__,
                   'parameters_json': json.dumps({'codes': codes, **kwargs}, sort_keys=True),
                   'sdk_version': client.sdk_version, 'mcp_server_sha256': client.mcp_server_sha256,
                   'requested_at': utcnow(), 'success': False, 'retry_count': 0}
            ledger.append(row)
            write_json(OUT / 'raw_request_manifest.json', ledger)
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    value = method(codes, **kwargs)
                frame = flatten(value)
                path = OUT / (name + '.parquet')
                frame.to_parquet(path, index=False)
                row.update(completed_at=utcnow(), response_path=path.name,
                           response_sha256=fingerprint(path), row_count=len(frame), success=True, error=None)
                return frame
            except BaseException as exc:
                message = str(exc).lower()
                row.update(completed_at=utcnow(), error=type(exc).__name__,
                           error_flags={key: key in message for key in
                                        ['permission denied', 'unable to open', 'unable to create',
                                         'file signature not found', 'lock', 'errno = 13']})
                raise
            finally:
                write_json(OUT / 'raw_request_manifest.json', ledger)
                print('REQUEST', name, row['success'], flush=True)

        base = pd.read_parquet(OLD / 'development_candidate_inputs.parquet')
        dates = pd.DatetimeIndex(sorted(base.date.unique()))
        if dates.max() != pd.Timestamp('2024-08-23') or dates.min() != pd.Timestamp('2014-01-02'):
            raise ValueError('Development bounds changed')
        raw_index = request('index500', client._info.get_index_constituent, ['000905.SH'], is_local=False)
        idx = expand_intervals(raw_index, dates, 'CON_CODE', 'INDATE', 'OUTDATE')
        if not idx.groupby('date').size().reindex(dates).eq(500).all():
            raise ValueError('Index count mismatch')
        raw_tech = request('technology', client._info.get_industry_constituent,
                           ['801080.SI', '801750.SI', '801770.SI'], is_local=False)
        code_col = next(c for c in ['CON_CODE', 'MARKET_CODE', 'STOCK_CODE'] if c in raw_tech)
        raw_tech['industry_code'] = raw_tech['_response_key']
        tech = expand_intervals(raw_tech, dates, code_col, 'INDATE', 'OUTDATE', ('industry_code',))
        new_members = additions(idx, tech, base[base.pool_origin.eq('original_csi300')])
        old_members = base[base.pool_origin.eq('added_csi500_tech')]
        keys = ['date', 'code', 'industry_code']
        pd.testing.assert_frame_equal(new_members[keys].sort_values(keys).reset_index(drop=True),
            old_members[keys].sort_values(keys).reset_index(drop=True), check_dtype=False)
        codes = sorted(old_members.code.unique())
        bars, statuses = [], []
        row_lineage = []
        for offset in range(0, len(codes), 25):
            batch = codes[offset:offset+25]
            args = {'begin_date': 20140102, 'end_date': 20240823, 'is_local': False}
            qname, sname = f'bars_{offset:04d}', f'status_{offset:04d}'
            bar = request(qname, client._market.query_kline, batch,
                          period=client.ad.constant.Period.day.value, **args)
            if 'code' not in bar:
                bar['code'] = bar['_response_key']
            stat = request(sname, client._info.get_history_stock_status, batch, **args)
            bars.append(bar)
            statuses.append(stat)
            for code in batch:
                row_lineage.append({'code': code, 'quote_request_id': qname, 'status_request_id': sname,
                                    'index_request_id': 'index500', 'industry_request_id': 'technology'})
        returns = normalize_returns(pd.concat(bars, ignore_index=True), pd.concat(statuses, ignore_index=True), dates)
        paired = old_members.merge(returns, on=['date', 'code'], how='left', suffixes=('_old', '_new'), validate='one_to_one')
        np.testing.assert_allclose(paired.r_1d_old, paired.r_1d_new, equal_nan=True, rtol=0, atol=1e-12)
        lineage = old_members[['date', 'code']].merge(pd.DataFrame(row_lineage), on='code', validate='many_to_one')
        lineage.to_parquet(OUT / 'reference_row_lineage.parquet', index=False)
        report.update(status='REFERENCE_SOURCE_REVERIFIED', original_values_and_membership_match=True,
                      rows=len(paired), request_count=len(ledger), sdk_version=client.sdk_version,
                      request_manifest_sha256=fingerprint(OUT / 'raw_request_manifest.json'),
                      original_input_sha256=fingerprint(OLD / 'development_candidate_inputs.parquet'),
                      row_lineage_sha256=fingerprint(OUT / 'reference_row_lineage.parquet'),
                      native_announcement_verified=False, timing_policy='D05_D06_DERIVED_POLICY',
                      note='Fresh requests verify equivalent Development inputs; these are not reconstructed old request times.')
    except BaseException as exc:
        report.update(status='SOURCE_REVERIFICATION_FAILED', error_type=type(exc).__name__)
    finally:
        if client is not None:
            try:
                client.logout()
            except BaseException:
                pass
        report.update(completed_at=utcnow(), code_sha256=fingerprint(Path(__file__)))
        write_json(OUT / 'report.json', report)
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
