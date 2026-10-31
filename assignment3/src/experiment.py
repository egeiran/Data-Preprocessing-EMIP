"""One fixed development-only experiment, with no hyperparameter search."""
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from time import perf_counter

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

from data import A3, ROOT, FEATURE_GROUPS, read_config, sha256, write_json
from checks import run_checks


def scores(truth, predicted):
    # Per-person scores average over classes actually observed for that person.
    # A one-class person's result is conditional performance, not two-class discrimination.
    labels = np.unique(truth)
    return {"balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
            "macro_f1": float(f1_score(truth, predicted, labels=labels,
                                        average="macro", zero_division=0))}


def main():
    config = read_config()
    run_checks()
    output = A3 / "results/first_validation"
    output.mkdir(parents=True, exist_ok=True)
    # Explicit read filter: locked people are never supplied to any model.
    data = pd.read_parquet(A3 / "cache/windows.parquet", filters=[("split", "==", "development")])
    test_ids = set(config["a1_test_participants"])
    assert not (set(data.pid) & test_ids)
    people = np.array(sorted(data.pid.unique(), key=int))
    splitter = KFold(n_splits=config["validation_folds"], shuffle=True, random_state=config["seed"])
    definitions = [
        ("dummy_most_frequent", "B_without_absolute_position"),
        ("tree_A_with_position", "A_with_position"),
        ("tree_B_without_absolute_position", "B_without_absolute_position"),
    ]
    folds, fold_rows, person_rows, prediction_rows = [], [], [], []
    for number, (train_ids, validation_ids) in enumerate(splitter.split(people), start=1):
        train_people, validation_people = people[train_ids], people[validation_ids]
        assert not (set(train_people) & set(validation_people))
        assert not ((set(train_people) | set(validation_people)) & test_ids)
        train = data[data.pid.isin(train_people)]
        validation = data[data.pid.isin(validation_people)]
        assert set(train.target) == {0, 1} and set(validation.target) == {0, 1}
        folds.append({"fold": number, "train_participants": train_people.tolist(),
                      "validation_participants": validation_people.tolist(),
                      "train_windows": len(train), "validation_windows": len(validation)})
        for name, group in definitions:
            features = FEATURE_GROUPS[group]
            Xtrain, Xval = train[features].to_numpy(dtype=np.float64), validation[features].to_numpy(dtype=np.float64)
            ytrain, yval = train.target.to_numpy(), validation.target.to_numpy()
            model = (DummyClassifier(strategy="most_frequent") if name.startswith("dummy")
                     else DecisionTreeClassifier(**config["tree"], random_state=config["seed"]))
            start = perf_counter()
            model.fit(Xtrain, ytrain)
            fit_s = perf_counter() - start
            start = perf_counter()
            predicted = model.predict(Xval)
            predict_s = perf_counter() - start
            fold_rows.append({"model": name, "feature_group": group, "fold": number,
                              "train_windows": len(train), "validation_windows": len(validation),
                              **scores(yval, predicted), "fit_seconds": fit_s,
                              "predict_seconds": predict_s,
                              "predict_microseconds_per_window": predict_s / len(validation) * 1e6,
                              "tree_depth": model.get_depth() if hasattr(model, "get_depth") else 0,
                              "tree_nodes": model.tree_.node_count if hasattr(model, "tree_") else 0})
            predictions = validation[["pid", "segment", "window_index", "target"]].copy()
            predictions["predicted"] = predicted
            predictions["model"], predictions["fold"] = name, number
            prediction_rows.append(predictions)
            for pid, part in predictions.groupby("pid", sort=False):
                person_rows.append({"model": name, "fold": number, "pid": pid,
                                    "windows": len(part), "observed_classes": int(part.target.nunique()),
                                    "rectangle_windows": int((part.target == 0).sum()),
                                    "vehicle_windows": int((part.target == 1).sum()),
                                    **scores(part.target, part.predicted)})
            print(f"Fold {number} {name}: {fold_rows[-1]}", flush=True)
    metrics = pd.DataFrame(fold_rows)
    participants = pd.DataFrame(person_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    assert not (set(predictions.pid) & test_ids)
    assert len(participants) == len(people) * len(definitions)
    assert not participants.duplicated(["model", "pid"]).any()
    assert not predictions.duplicated(["model", "pid", "segment", "window_index"]).any()
    summary = {}
    for name, _ in definitions:
        fold = metrics[metrics.model == name]
        person = participants[participants.model == name]
        summary[name] = {"fold_mean": {}, "fold_std_sample": {}, "fold_min": {}, "fold_max": {},
                         "equal_participant_mean": {}, "equal_participant_std_sample": {}}
        for metric in ["balanced_accuracy", "macro_f1"]:
            for key, value in [("fold_mean", fold[metric].mean()),
                               ("fold_std_sample", fold[metric].std(ddof=1)),
                               ("fold_min", fold[metric].min()), ("fold_max", fold[metric].max()),
                               ("equal_participant_mean", person[metric].mean()),
                               ("equal_participant_std_sample", person[metric].std(ddof=1))]:
                summary[name][key][metric] = float(value)
        summary[name]["total_fit_seconds"] = float(fold.fit_seconds.sum())
        summary[name]["total_predict_seconds"] = float(fold.predict_seconds.sum())
        own_predictions = predictions[predictions.model == name]
        summary[name]["pooled_out_of_fold"] = scores(own_predictions.target, own_predictions.predicted)
    metrics.to_csv(output / "fold_metrics.csv", index=False)
    participants.to_csv(output / "participant_metrics.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    write_json(output / "summary.json", summary)
    write_json(output / "folds.json", folds)
    write_json(output / "config.json", {**config, "feature_groups": FEATURE_GROUPS,
                                       "fold_method": "seeded KFold on sorted participant IDs, then all their windows",
                                       "models": definitions, "learned_transforms": "none"})
    packages = {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "pyarrow", "openpyxl"]}
    environment = {"created_utc": datetime.now(timezone.utc).isoformat(),
                   "command": ".venv/bin/python -B assignment3/src/experiment.py",
                   "python": sys.version, "platform": platform.platform(),
                   "cpu": platform.processor(), "logical_cpus": os.cpu_count(), "packages": packages,
                   "thread_environment": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]},
                   "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
                   "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT).decode(),
                   "windows_sha256": sha256(A3 / "cache/windows.parquet"),
                   "data_provenance_sha256": sha256(A3 / "results/data/provenance.json"),
                   "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted((A3 / "src").glob("*.py"))},
                   "timing": "perf_counter wall time around one fit and one predict per fold; arrays prepared first; no I/O, feature building or metric calculations included; small timings are noisy",
                   "per_person_metrics": "average over observed true classes; report class support; one-class people do not test two-class discrimination; equal-participant summary weights every development person once",
                   "test_evaluation": "not performed"}
    write_json(output / "environment.json", environment)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
