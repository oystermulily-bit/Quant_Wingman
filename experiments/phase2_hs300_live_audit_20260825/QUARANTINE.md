# QUARANTINED_REFERENCE_ONLY

Path:

`D:/Hulucoding/AmAzing_Data/output/hs300_daily_2013_present/`

Status at audit time 2026-08-25:

- Directory does not exist on disk.
- 2026-08-12 build log and 2026-08-20 preflight still describe it.
- Do not recreate, patch, or copy it into the formal snapshot.
- Do not delete this note if the old directory reappears; keep it quarantined.

Allowed:

- Compare schema and approximate scale from the 2026-08-12 build log and preflight hashes.
- Detect order-of-magnitude errors in a new build.

Forbidden:

- Feature input, labels, training, validation, backtest.
- Filling holes in the new snapshot.
- Historical constituent ground truth.
- Adjustment-price source.
- Using the weight-table `CLOSE` column from AmazingData as a substitute for missing 2010–2012 OHLCV.
