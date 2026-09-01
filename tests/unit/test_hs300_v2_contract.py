from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_pipeline.hs300.adapter import HS300WingmanAdapter
from data_pipeline.hs300.features_wingman import _cs_average_rank, pd_rank
from data_pipeline.hs300.industry import build_industry_membership
from data_pipeline.hs300.lineage import LINEAGE_COLUMNS, attach_lineage
from data_pipeline.hs300.seal import (
    assert_holdout_sealed,
    split_development_and_holdout,
    write_sealed_holdout,
)
from data_pipeline.hs300.time_policy import to_trade_date
from data_pipeline.hs300_panel import (
    HS300PanelDataManager,
    SnapshotManifestValidator,
)
from tests.unit.test_stage3_research import _make_snapshot


def test_to_trade_date_parses_mixed_sentinels_per_element() -> None:
    parsed = to_trade_date([20140102, 0, np.nan, 99991231, "2014-01-03", 18991231])
    assert parsed.iloc[0] == pd.Timestamp("2014-01-02")
    assert pd.isna(parsed.iloc[1])
    assert pd.isna(parsed.iloc[2])
    assert pd.isna(parsed.iloc[3])
    assert parsed.iloc[4] == pd.Timestamp("2014-01-03")
    assert pd.isna(parsed.iloc[5])


def test_industry_keeps_one_l1_on_conflict_and_closes_gaps() -> None:
    base = pd.DataFrame(
        {
            "INDEX_CODE": ["801180.SI", "801010.SI"],
            "LEVEL_TYPE": [1, 1],
            "LEVEL1_NAME": ["房地产", "农林牧渔"],
        }
    )
    constituents = pd.DataFrame(
        {
            "INDEX_CODE": ["801180.SI", "801010.SI", "801180.SI"],
            "CON_CODE": ["000002.SZ", "000002.SZ", "000002.SZ"],
            "INDATE": [20100104, 20100104, 20150105],
            "OUTDATE": [20120601, 0, 0],
        }
    )
    membership, exceptions = build_industry_membership(
        constituents, base, ["000002.SZ"], expected_l1_count=2
    )
    first = membership.sort_values("valid_from").iloc[0]
    assert first["industry_code"] == "801010.SI"
    assert first["valid_to"] == pd.Timestamp("2015-01-04")
    assert (exceptions["reason"] == "CONFLICTING_L1_DROPPED_SECONDARY").any()


def test_cs_rank_uses_average_rank_for_ties() -> None:
    values = np.array([1.0, 2.0, 2.0, 4.0], dtype=np.float32)
    expected = pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=np.float32)
    np.testing.assert_allclose(pd_rank(values), expected)
    ranked = _cs_average_rank(
        np.array([[1.0], [2.0], [2.0], [4.0]], dtype=np.float32),
        np.ones((4, 1), dtype=bool),
    )
    np.testing.assert_allclose(ranked[:, 0], expected)


def test_holdout_prices_are_physically_split(tmp_path: Path) -> None:
    dates = pd.to_datetime(["2024-08-23", "2024-08-26"])
    frame = pd.DataFrame(
        {"date": dates, "code": ["000001.SZ", "000001.SZ"], "close_raw": [10.0, 11.0]}
    )
    development, holdout = split_development_and_holdout(frame, pd.Timestamp("2024-08-26"))
    assert list(development["date"]) == [pd.Timestamp("2024-08-23")]
    std = tmp_path / "standardized"
    std.mkdir()
    development.to_parquet(std / "daily_bars.parquet", index=False)
    development.assign(is_member=True).to_parquet(std / "universe_membership.parquet", index=False)
    development.to_parquet(std / "trading_status.parquet", index=False)
    write_sealed_holdout(
        tmp_path,
        {"daily_bars": holdout},
        holdout_start="2024-08-26",
        holdout_date_count=1,
        holdout_date_hash="abc",
        protocol="test",
    )
    assert_holdout_sealed(tmp_path, pd.Timestamp("2024-08-26"))
    sealed = tmp_path / "sealed" / "holdout" / "daily_bars.parquet"
    assert sealed.is_file()
    stored = pd.read_parquet(tmp_path / "standardized" / "daily_bars.parquet")
    assert pd.Timestamp("2024-08-26") not in set(pd.to_datetime(stored["date"]))
    adapter = HS300WingmanAdapter(tmp_path)
    with pytest.raises(PermissionError, match="sealed"):
        adapter.load_holdout_prices(capability_token="nope")
    token = (tmp_path / "sealed" / "holdout.capability").read_text(encoding="utf-8").strip()
    loaded = adapter.load_holdout_prices(capability_token=token)
    assert list(loaded["daily_bars"]["date"]) == [pd.Timestamp("2024-08-26")]
    with pytest.raises(PermissionError, match="consumed"):
        adapter.load_holdout_prices(capability_token=token)
    lock = json.loads((tmp_path / "sealed" / "holdout.lock.json").read_text(encoding="utf-8"))
    assert "unseal_phrase" not in lock
    log = (tmp_path / "sealed" / "holdout.access.log").read_text(encoding="utf-8")
    assert "granted" in log
    assert "denied" in log


def test_attach_lineage_writes_required_fields() -> None:
    frame = attach_lineage(
        pd.DataFrame({"code": ["000001.SZ"]}),
        schema_version="hs300_pit_v2",
        snapshot_id="csi300_2014_present_v2",
        data_version="v2:2026-08-24",
        source="AmazingData SDK 1.1.9",
        fetched_at="2026-08-25T00:00:00+00:00",
    )
    for column in LINEAGE_COLUMNS:
        assert column in frame.columns


def test_row_count_contract_rejects_mismatch(tmp_path: Path) -> None:
    root = _make_snapshot(tmp_path / "snapshot", dates=20, symbols=3)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"]["daily_bars"]["rows"] = 0
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _, report = SnapshotManifestValidator(root).validate()
    assert not report.passed
    assert any(issue.code == "ROW_COUNT_CONTRACT" for issue in report.issues)


def test_industry_coverage_below_threshold_is_error(tmp_path: Path) -> None:
    root = _make_snapshot(tmp_path / "snapshot", dates=20, symbols=3)
    industries = pd.read_parquet(root / "industry_membership.parquet")
    industries["valid_from"] = pd.Timestamp("2099-01-01")
    industries["valid_to"] = pd.NaT
    industries.to_parquet(root / "industry_membership.parquet", index=False)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    import hashlib

    digest = hashlib.sha256((root / "industry_membership.parquet").read_bytes()).hexdigest()
    manifest["files"]["industry_membership"]["sha256"] = digest
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    manager = HS300PanelDataManager(root, expected_members=3, required_start=None)
    manager.load(raise_on_gate_failure=False)
    assert manager.report is not None
    assert not manager.report.passed
    assert any(
        issue.code == "INDUSTRY_COVERAGE_BELOW_THRESHOLD" for issue in manager.report.issues
    )
