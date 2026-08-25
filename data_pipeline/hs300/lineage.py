"""Lineage columns required on every standardized table."""

from __future__ import annotations

from typing import Any

import pandas as pd


LINEAGE_COLUMNS = (
    "schema_version",
    "snapshot_id",
    "data_version",
    "source",
    "fetched_at",
)


def attach_lineage(
    frame: pd.DataFrame,
    *,
    schema_version: str,
    snapshot_id: str,
    data_version: str,
    source: str,
    fetched_at: str,
) -> pd.DataFrame:
    out = frame.copy()
    out["schema_version"] = schema_version
    out["snapshot_id"] = snapshot_id
    out["data_version"] = data_version
    out["source"] = source
    out["fetched_at"] = fetched_at
    return out


def lineage_payload(
    *,
    schema_version: str,
    snapshot_id: str,
    data_version: str,
    source: str,
    fetched_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "snapshot_id": snapshot_id,
        "data_version": data_version,
        "source": source,
        "fetched_at": fetched_at,
    }
