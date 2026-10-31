# Assignment 3: known code-task identification from eye tracking

Start with [WORKLOG.md](WORKLOG.md): agreed scope, decisions, evidence, limitations
and handover. [Official assignment](assignment.pdf). A1 files stay at the root.

## Setup

Run from the repository root. The existing `.venv` already contains the required
packages. For a fresh environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r assignment3/results/first_validation/requirements.txt
```

Python 3.12 was used. A3 currently needs numpy, pandas, scikit-learn, pyarrow and
openpyxl. The recorded requirements reproduce the installed environment; root
`requirements.txt` remains the original, less tightly pinned A1 setup.
Exact versions and hashes are saved with each run. Raw files must be in
`emip_dataset/rawdata/`, and the existing A1 `dataset.parquet` must be present.
If missing, recreate A1 separately with `.venv/bin/python cleaning.py`; this writes
the root cache. A3 never silently rebuilds it.

## First experiment

```bash
.venv/bin/python -B assignment3/src/data.py
.venv/bin/python -B assignment3/src/checks.py
.venv/bin/python -B assignment3/src/experiment.py
```

The builder reads cache signals and raw screen boundaries. It writes ignored
`assignment3/cache/windows.parquet` and small audit files in `assignment3/results/data/`.
The experiment reads development windows only and writes fold, participant and
prediction results plus configuration/environment to `assignment3/results/first_validation/`.
The seven A1 test people remain locked and are not scored. Re-running overwrites
these A3 output paths; copy a run directory before changing the configuration.

`config.json` holds the fixed choices. No scaling, PCA, tuning, ensembles or deep
models are run. Large generated data and fitted models must stay out of Git.

The first run produced 1,366 windows: 992 development and 374 locked-test windows.
Three-fold development balanced accuracy was 0.500 for dummy, 0.611 for tree A
and 0.507 for tree B. See the worklog for uncertainty, per-person results,
coverage/interpolation limits and remaining assignment requirements.
