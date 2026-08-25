from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.hs300.raw_store import RawStore


def test_raw_store_does_not_overwrite_success(tmp_path: Path) -> None:
    store = RawStore(tmp_path)
    first = pd.DataFrame({"CON_CODE": ["000001.SZ"], "WEIGHT": [1.0]})
    second = pd.DataFrame({"CON_CODE": ["000002.SZ"], "WEIGHT": [2.0]})
    meta1 = store.save(
        method="index_weight",
        part_id="year=2013",
        frame=first,
        parameters={"year": 2013},
        mcp_server_sha256="ABC",
        sdk_version="1.1.9",
    )
    meta2 = store.save(
        method="index_weight",
        part_id="year=2013",
        frame=second,
        parameters={"year": 2013},
        mcp_server_sha256="ABC",
        sdk_version="1.1.9",
    )
    assert meta1["row_count"] == 1
    assert meta2.get("skipped") is True
    loaded = pd.read_parquet(store.part_paths("index_weight", "year=2013")[0])
    assert list(loaded["CON_CODE"]) == ["000001.SZ"]
