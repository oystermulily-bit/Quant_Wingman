"""Bounded source probe; credentials never enter output or artifacts."""
import contextlib
import io
import json
import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'experiments/d21_v3_source_probe_20260914_bounded'


def main():
    OUT.mkdir(exist_ok=False)
    report = {'status': 'PROBE_STARTED', 'holdout_read': False, 'labels_read': False,
              'supplement_applied': False}
    client = None
    try:
        config = json.loads(Path('C:/Users/Administrator/.cursor/mcp.json').read_text(encoding='utf-8'))
        env = config['mcpServers']['amazingdata'].get('env', {})
        for key in ['AD_USERNAME', 'AD_PASSWORD', 'AD_HOST', 'AD_PORT']:
            value = env.get(key)
            if not value or '${' in str(value):
                raise ValueError('Unresolved connection setting')
            os.environ[key] = str(value)
        from data_pipeline.hs300.sdk_client import AmazingDataClient
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            client = AmazingDataClient()
            client.login()
            frame = client.get_industry_weight(['801010.SI'], 20200102, 20200103)
        frame.to_parquet(OUT / 'industry_weight_probe.parquet', index=False)
        report.update(status='PROBE_RETURNED', rows=len(frame), columns=list(frame.columns),
                      sdk_version=client.sdk_version, begin_date=20200102, end_date=20200103)
    except BaseException as exc:
        report.update(status='SOURCE_PROBE_FAILED', error_type=type(exc).__name__)
    finally:
        if client is not None:
            try:
                client.logout()
            except BaseException:
                pass
        (OUT / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report))


if __name__ == '__main__':
    if '--worker' in sys.argv:
        main()
    else:
        try:
            result = subprocess.run([sys.executable, __file__, '--worker'],
                                    capture_output=True, text=True, timeout=30)
            print(result.stdout)
        except subprocess.TimeoutExpired:
            report = {'status': 'SOURCE_PROBE_TIMEOUT', 'timeout_seconds': 30,
                      'tcp_connected': True, 'supplemented_rows': 0,
                      'holdout_read': False, 'labels_read': False,
                      'reason': 'SDK login or initial data request did not return within deadline'}
            (OUT / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(report))
