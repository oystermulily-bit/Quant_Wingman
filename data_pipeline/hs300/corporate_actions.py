"""Corporate action ledger from frozen AmazingData event tables."""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from .time_policy import to_trade_date


def _first_column(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.Series | None:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return None


def _payload_hash(row: pd.Series) -> str:
    payload = json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _known_at(announcement) -> tuple[pd.Series, pd.Series]:
    parsed = pd.to_datetime(announcement, errors="coerce")
    source = pd.Series(
        ["NATIVE" if pd.notna(value) else "MISSING" for value in parsed],
        index=parsed.index,
        dtype="string",
    )
    return parsed, source


def build_corporate_actions(
    dividends: pd.DataFrame | None = None,
    rights: pd.DataFrame | None = None,
    equity: pd.DataFrame | None = None,
) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []
    if dividends is not None and not dividends.empty:
        code = _first_column(dividends, ("MARKET_CODE", "code"))
        ann = _first_column(dividends, ("DATE_DVD_ANN", "ANN_DATE", "announcement_at"))
        ex_date = _first_column(dividends, ("DATE_EX", "EX_DATE", "ex_date"))
        cash = _first_column(
            dividends,
            ("DVD_PER_SHARE_PRE_TAX_CASH", "CASH_DIVIDEND", "cash_dividend"),
        )
        stock = _first_column(
            dividends, ("DVD_PER_SHARE_STOCK", "STOCK_DIVIDEND_RATIO", "stock_dividend_ratio")
        )
        known_at, known_source = _known_at(ann)
        block = pd.DataFrame(
            {
                "code": code.astype(str).str.upper(),
                "action_type": "DIVIDEND",
                "announcement_at": known_at,
                "record_date": to_trade_date(_first_column(dividends, ("DATE_RECORD", "record_date")))
                if _first_column(dividends, ("DATE_RECORD", "record_date")) is not None
                else pd.NaT,
                "ex_date": to_trade_date(ex_date) if ex_date is not None else pd.NaT,
                "payment_date": to_trade_date(
                    _first_column(dividends, ("DATE_PAY", "payment_date"))
                )
                if _first_column(dividends, ("DATE_PAY", "payment_date")) is not None
                else pd.NaT,
                "effective_at": to_trade_date(ex_date) if ex_date is not None else pd.NaT,
                "cash_dividend": pd.to_numeric(cash, errors="coerce") if cash is not None else pd.NA,
                "stock_dividend_ratio": pd.to_numeric(stock, errors="coerce")
                if stock is not None
                else pd.NA,
                "rights_ratio": pd.NA,
                "rights_price": pd.NA,
                "known_at": known_at,
                "known_at_source": known_source,
            }
        )
        block["raw_payload_hash"] = dividends.apply(_payload_hash, axis=1)
        blocks.append(block)
    if rights is not None and not rights.empty:
        code = _first_column(rights, ("MARKET_CODE", "code"))
        ann = _first_column(rights, ("ANN_DATE", "EXECUTE_DATE", "announcement_at"))
        # EXECUTE_DATE is allowed only when it is the announcement field for rights.
        known_at, known_source = _known_at(ann)
        ratio = _first_column(rights, ("RIGHTS_RATIO", "RATIO", "rights_ratio"))
        price = _first_column(rights, ("RIGHTS_PRICE", "PRICE", "rights_price"))
        ex_date = _first_column(rights, ("DATE_EX", "EX_DATE", "ex_date"))
        block = pd.DataFrame(
            {
                "code": code.astype(str).str.upper(),
                "action_type": "RIGHT_ISSUE",
                "announcement_at": known_at,
                "record_date": pd.NaT,
                "ex_date": to_trade_date(ex_date) if ex_date is not None else pd.NaT,
                "payment_date": pd.NaT,
                "effective_at": to_trade_date(ex_date) if ex_date is not None else pd.NaT,
                "cash_dividend": pd.NA,
                "stock_dividend_ratio": pd.NA,
                "rights_ratio": pd.to_numeric(ratio, errors="coerce") if ratio is not None else pd.NA,
                "rights_price": pd.to_numeric(price, errors="coerce") if price is not None else pd.NA,
                "known_at": known_at,
                "known_at_source": known_source,
            }
        )
        block["raw_payload_hash"] = rights.apply(_payload_hash, axis=1)
        blocks.append(block)
    if equity is not None and not equity.empty:
        code = _first_column(equity, ("MARKET_CODE", "code"))
        ann = _first_column(equity, ("ANN_DATE", "announcement_at"))
        change = _first_column(equity, ("CHANGE_DATE", "effective_at"))
        known_at, known_source = _known_at(ann)
        block = pd.DataFrame(
            {
                "code": code.astype(str).str.upper() if code is not None else pd.NA,
                "action_type": "EQUITY_CHANGE",
                "announcement_at": known_at,
                "record_date": pd.NaT,
                "ex_date": pd.NaT,
                "payment_date": pd.NaT,
                "effective_at": to_trade_date(change) if change is not None else pd.NaT,
                "cash_dividend": pd.NA,
                "stock_dividend_ratio": pd.NA,
                "rights_ratio": pd.NA,
                "rights_price": pd.NA,
                "known_at": known_at,
                "known_at_source": known_source,
            }
        )
        block["raw_payload_hash"] = equity.apply(_payload_hash, axis=1)
        blocks.append(block)

    if not blocks:
        columns = [
            "action_id", "code", "action_type", "announcement_at", "record_date",
            "ex_date", "payment_date", "effective_at", "cash_dividend",
            "stock_dividend_ratio", "rights_ratio", "rights_price", "known_at",
            "known_at_source", "raw_payload_hash",
        ]
        return pd.DataFrame(columns=columns)

    out = pd.concat(blocks, ignore_index=True)
    out["action_id"] = [
        hashlib.sha256(f"{row.action_type}:{row.code}:{row.raw_payload_hash}".encode()).hexdigest()
        for row in out.itertuples()
    ]
    return out
