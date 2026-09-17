# IT3212 Assignment 1: Data Preprocessing

Preprocessing of the EMIP eye-tracking dataset. The report is `report.md` (submitted as PDF);
this repository holds the code that produced every number and figure in it.

## Setup

```
uv venv && uv pip install -r requirements.txt
```

Put the raw recordings in `emip_dataset/rawdata/` (`<id>_rawdata.tsv`, plus `211_rawdata.xlsx`).
The data is not in the repository.

## Run order

| script | report section | what it does |
|---|---|---|
| `explore.py` | 1 | first rows, dtypes, summary statistics, unique values, redundancy |
| `cleaning.py` | 2 | validity rules, hole classification, interpolation, exclusion; caches `dataset.parquet` |
| `outliers.py` | 3 | velocity limit, pupil Z-score and IQR, removal and capping |
| `transform.py` | 4, 5, 6 | encoding, grouped split, scaling, PCA |
| `figures.py` | all | writes `figures/fig1..fig6.png` |
| `trym_review.py` | 4, 5, 6 | independent validation of the split, scaling and PCA; reads the cache, writes nothing |

`trym_review.py` is the exception: it requires an existing `dataset.parquet` and refuses to
build one, so run `cleaning.py` first.

Each script imports the previous ones, so any of them can be run on its own with
`.venv/bin/python <script>.py`. The first run parses all 33 files (about 30 s); later runs
read the parquet cache. Delete `dataset.parquet` or call `build_dataset(rebuild=True)` after
changing `cleaning.py`.

`results.md` is the working log kept while the analysis was developed. Numbers in the report
take precedence where the two differ.
