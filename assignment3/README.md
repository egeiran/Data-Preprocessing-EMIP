# Assignment 3: which code is read, with and without gaze position

Main question: can eye tracking tell which of two code snippets an unseen person
reads, and does it still work when absolute gaze position is removed, a common
privacy safeguard? See *Problem and scope* in the worklog.

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

## Quality review and bounded model comparison

V1 results above are preserved. V2 adds at least 80% directly measured time to
the original rules; see `quality_policy.json`, `comparison_plan.json` and WORKLOG.
V2 has 961 development windows from the same 23 people, plus 369 locked-test
windows. No test model results are used to set the policy.

```bash
.venv/bin/python -B assignment3/src/quality.py
.venv/bin/python -B assignment3/src/quality.py --apply
.venv/bin/python -B assignment3/src/compare.py
```

Quality evidence is in `results/quality/`; V2 features are in ignored
`cache/windows_v2.parquet`. The first command audits development quality; the
second applies the declared policy and records all selection counts.

The comparison uses three nested participant folds, six settings each for tree,
kNN and RBF-SVM on both feature groups, plus untuned dummy/position controls.
Primary scoring gives each person equal weight. Scaling is fitted within training
pipelines only. Results, class/person errors, search scores, repeated timings and
environment are in `results/model_comparison_v2/`; start with `summary.csv`.
No final-test predictions or full-development final model are produced.

Current person-average balanced accuracy: tree A/B .616/.527, kNN A/B .584/.498,
SVM A/B .582/.534, dummy .500 and position-only stump .581. Read WORKLOG before
interpreting these small development results. Reruns overwrite the same version's
output directory; create a new version before changing analysis choices.

## V3 event features

```bash
.venv/bin/python -B assignment3/src/fixations.py
```

Adds eight I-VT fixation/saccade features to the same V2 windows, with thresholds
fixed in `feature_plan_v3.json`. Writes ignored `cache/windows_v3.parquet` and
evidence in `results/features_v3/`. Parquet bytes differ between machines, so V3
code checks V2 by keys and records portable content fingerprints (see WORKLOG).
`uv run python -B …` works the same as `.venv/bin/python -B …`.

## V3 models and ensembles

```bash
.venv/bin/python -B assignment3/src/models.py basic
.venv/bin/python -B assignment3/src/models.py ensembles
```

Follows `model_plan_v3.json`: tree/kNN/RBF-SVM tuned jointly with a
mutual-information filter, then AdaBoost (trees) and bagged kNN, all on A3/B3 with
the V2 folds. Each part writes its own directory under `results/models_v3/`.

## Sequence CNN and transfer

```bash
.venv/bin/python -B assignment3/src/sequences.py
.venv/bin/python -B assignment3/src/cnn.py
```

Follows `deep_plan.json`: 500-step gaze sequences (ignored `cache/sequences.npz`),
a small 1D CNN tuned with nested person folds, and transfer from instruction vs
answer screens. Needs PyTorch (CPU build; on Linux use the CPU wheel index to avoid
CUDA downloads). The run takes several hours on a laptop CPU.

## Locked test, figures and report

```bash
.venv/bin/python -B assignment3/src/final_test.py      # one time only; refuses to rerun
.venv/bin/python -B assignment3/src/figures.py
.venv/bin/python -B assignment3/src/export_report.py   # report.md -> report.html/.pdf
```

`final_test.py` follows `final_test_plan.json`: every method is tuned on the 23
development people, refit on all development windows and scored once on the seven
locked people (results in `results/final_test/`). The report export uses the A1
stylesheet and headless Chrome.
