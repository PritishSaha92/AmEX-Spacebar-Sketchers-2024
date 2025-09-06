# data/

This directory stores datasets.

Git policy:
- `data/raw/` — raw competition files are tracked.
- `data/interim/` — intermediate/prepared files are not tracked.
- `data/processed/` — post-processed files are not tracked.

Place the following raw files in `data/raw/` as applicable:
- `add_event.parquet`
- `add_trans.parquet`
- `offer_metadata.parquet`
- `train_data.parquet`
- `test_data.parquet`
- `test_data_r3.parquet`
- `submission_template.csv`

Interim artifacts produced by preprocessing/feature engineering will appear under `data/interim/` (e.g., `train_1_prepared.parquet`, `valid_1_prepared.parquet`, `test_1_prepared.parquet`) and are ignored by git.


