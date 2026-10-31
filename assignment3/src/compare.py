"""Bounded nested participant validation. Never evaluate the locked final test."""
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from time import perf_counter

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data import A3, ROOT, FEATURE_GROUPS, POSITION, sha256, write_json
from checks import run_checks


def metrics(truth, predicted):
    """Two-class metrics are undefined for a person with only one observed class."""
    matrix = np.zeros((2, 2), dtype=int)
    np.add.at(matrix, (np.asarray(truth, dtype=int), np.asarray(predicted, dtype=int)), 1)
    support = matrix.sum(axis=1)
    recalls = np.divide(np.diag(matrix), support, out=np.zeros(2), where=support > 0)
    both = bool((support > 0).all())
    f1 = np.divide(2*np.diag(matrix), support+matrix.sum(axis=0),
                   out=np.zeros(2), where=support+matrix.sum(axis=0) > 0)
    return {"rectangle_windows": int(support[0]), "vehicle_windows": int(support[1]),
            "tn": int(matrix[0, 0]), "fp": int(matrix[0, 1]),
            "fn": int(matrix[1, 0]), "tp": int(matrix[1, 1]),
            "two_class_defined": both,
            "balanced_accuracy": float(recalls.mean()) if both else None,
            "macro_f1": float(f1.mean()) if both else None,
            "observed_class_recall": float(recalls[support > 0].mean()),
            "rectangle_recall": float(recalls[0]) if support[0] else None,
            "vehicle_recall": float(recalls[1]) if support[1] else None}


def participant_metrics(predictions):
    return pd.DataFrame([{"pid": pid, "windows": len(rows),
                          **metrics(rows.target, rows.predicted)}
                         for pid, rows in predictions.groupby("pid", sort=False)])


def estimator(family, parameters, seed):
    if family == "tree":
        return DecisionTreeClassifier(**parameters, class_weight="balanced", random_state=seed)
    if family == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(**parameters, p=2, n_jobs=1))
    if family == "svm_rbf":
        return make_pipeline(StandardScaler(), SVC(**parameters, kernel="rbf", class_weight="balanced", cache_size=128))
    if family == "dummy":
        return DummyClassifier(strategy="most_frequent")
    if family == "position_stump":
        return DecisionTreeClassifier(max_depth=1, min_samples_leaf=10, class_weight="balanced", random_state=seed)
    raise ValueError(family)


def fit_model(family, parameters, seed, train, features):
    model = estimator(family, parameters, seed)
    X, y = train[features].to_numpy(dtype=float), train.target.to_numpy()
    model.fit(X, y)
    if family in ("knn", "svm_rbf"):
        scaler = model.steps[0][1]
        assert scaler.n_samples_seen_ == len(train)
        assert np.allclose(scaler.mean_, X.mean(axis=0))
    return model


def inner_partitions(people, n_splits, seed):
    people = np.array(sorted(people, key=int))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [{"train": people[tr].tolist(), "validation": people[va].tolist()}
            for tr, va in splitter.split(people)]


def tune(train, partitions, family, features, grid, seed):
    results = []
    started = perf_counter()
    for parameters in ParameterGrid(grid):
        predictions = []
        for part in partitions:
            assert not (set(part["train"]) & set(part["validation"]))
            inner_train = train[train.pid.isin(part["train"])]
            validation = train[train.pid.isin(part["validation"])]
            model = fit_model(family, parameters, seed, inner_train, features)
            pred = validation[["pid", "target"]].copy()
            pred["predicted"] = model.predict(validation[features].to_numpy(dtype=float))
            predictions.append(pred)
        out_of_fold = pd.concat(predictions)
        assert len(out_of_fold) == len(train) and not out_of_fold.index.duplicated().any()
        people = participant_metrics(out_of_fold)
        score = people.balanced_accuracy.mean()
        if not np.isfinite(score):
            raise ValueError("No two-class participants for inner model selection")
        results.append({"parameters": parameters, "inner_equal_person_balanced_accuracy": float(score),
                        "eligible_people": int(people.two_class_defined.sum())})
    best = max(results, key=lambda row: row["inner_equal_person_balanced_accuracy"])
    return best["parameters"], results, perf_counter()-started


def repeated_time(model, Xtrain, ytrain, Xval, specification):
    """Warm up, then time fit/predict in batches; cloning is outside each fit timer."""
    clone(model).fit(Xtrain, ytrain)
    model.predict(Xval)
    rows = []
    for operation in ("fit", "predict"):
        for batch in range(specification["batches"]):
            elapsed, calls = 0.0, 0
            while elapsed < specification["min_seconds_per_batch"] and calls < specification["max_calls_per_batch"]:
                fresh = clone(model) if operation == "fit" else model
                started = perf_counter()
                if operation == "fit": fresh.fit(Xtrain, ytrain)
                else: fresh.predict(Xval)
                elapsed += perf_counter()-started
                calls += 1
            rows.append({"operation": operation, "batch": batch+1, "calls": calls,
                         "batch_seconds": elapsed, "seconds_per_call": elapsed/calls})
    return rows


def main():
    run_checks()
    plan_path = A3 / "comparison_plan.json"
    plan = json.loads(plan_path.read_text())
    selection = json.loads((A3 / "results/quality/selection.json").read_text())
    dataset = ROOT / plan["dataset"]
    assert sha256(dataset) == selection["v2_windows_sha256"]
    assert sha256(A3 / "src/quality.py") == selection["quality_source_sha256"]
    assert selection["policy"] == json.loads((A3 / "quality_policy.json").read_text())
    locked = set(selection["policy"]["test_participants"])
    data = pd.read_parquet(dataset, filters=[("split", "==", "development")])
    assert not (set(data.pid) & locked)
    assert data.measured_fraction.ge(selection["policy"]["min_measured_fraction"]).all()
    outer_folds = json.loads((ROOT / plan["outer_folds"]).read_text())
    validation_people = [pid for fold in outer_folds for pid in fold["validation_participants"]]
    assert len(validation_people) == len(set(validation_people))
    assert set(validation_people) == set(data.pid)
    # Explicit numeric allowlists. Quality/time/identity/labels never enter X.
    groups = {**FEATURE_GROUPS, "position_only": POSITION}
    forbidden = {"pid", "target", "stimulus", "segment", "window_index", "start_us", "end_us",
                 "split", "measured_fraction", "accepted_v2", "language", "file"}
    for names in groups.values():
        assert not (set(names) & forbidden) and np.isfinite(data[names]).all().all()
    # One-class scoring policy is deliberately different from the historical V1 helper.
    one_class = metrics(np.array([1, 1]), np.array([1, 0]))
    assert one_class["balanced_accuracy"] is None and one_class["macro_f1"] is None
    assert one_class["observed_class_recall"] == .5
    output = A3 / "results/model_comparison_v2"
    output.mkdir(exist_ok=True)
    definitions = [(family, group) for family in plan["grids"] for group in plan["feature_groups"]]
    definitions += [("dummy", "B_without_absolute_position"), ("position_stump", "position_only")]
    fold_rows, person_rows, predictions, searches, timing_rows, inner_folds = [], [], [], [], [], []
    for fold in outer_folds:
        number = fold["fold"]
        tr_ids, va_ids = set(fold["train_participants"]), set(fold["validation_participants"])
        assert not tr_ids & va_ids and not (tr_ids | va_ids) & locked
        assert tr_ids | va_ids == set(data.pid)
        train, validation = data[data.pid.isin(tr_ids)], data[data.pid.isin(va_ids)]
        partitions = inner_partitions(tr_ids, plan["inner_group_folds"], plan["seed"])
        for part in partitions:
            assert set(part["train"]) | set(part["validation"]) == tr_ids
            assert not (set(part["train"]) | set(part["validation"])) & va_ids
        inner_folds.append({"outer_fold": number, "partitions": partitions})
        for family, group in definitions:
            name = f"{family}__{group}"
            features = groups[group]
            parameters, search_seconds = {}, 0.0
            if family in plan["grids"]:
                parameters, candidates, search_seconds = tune(
                    train, partitions, family, features, plan["grids"][family], plan["seed"])
                for candidate in candidates:
                    searches.append({"model": name, "outer_fold": number,
                                     "parameters": json.dumps(candidate["parameters"], sort_keys=True),
                                     "inner_equal_person_balanced_accuracy": candidate["inner_equal_person_balanced_accuracy"],
                                     "eligible_people": candidate["eligible_people"]})
            model = fit_model(family, parameters, plan["seed"], train, features)
            Xtr, Xval = train[features].to_numpy(dtype=float), validation[features].to_numpy(dtype=float)
            pred = validation[["pid", "segment", "window_index", "target"]].copy()
            pred["predicted"] = model.predict(Xval)
            pred["model"], pred["fold"] = name, number
            predictions.append(pred)
            people = participant_metrics(pred)
            people["model"], people["fold"] = name, number
            person_rows.append(people)
            times = repeated_time(model, Xtr, train.target.to_numpy(), Xval, plan["timing"])
            for row in times: timing_rows.append({"model": name, "fold": number, **row})
            fit_times = [r["seconds_per_call"] for r in times if r["operation"] == "fit"]
            predict_times = [r["seconds_per_call"] for r in times if r["operation"] == "predict"]
            core = model.steps[-1][1] if hasattr(model, "steps") else model
            complexity = {"depth": int(core.get_depth()), "nodes": int(core.tree_.node_count)} if hasattr(core, "tree_") else {}
            if family == "position_stump":
                axis = int(core.tree_.feature[0])
                complexity.update({"split_feature": features[axis] if axis >= 0 else None,
                                   "threshold_px": float(core.tree_.threshold[0]) if axis >= 0 else None})
            if family == "svm_rbf": complexity = {"support_vectors": int(core.n_support_.sum())}
            if family == "knn": complexity = {"stored_training_windows": len(train), "k": parameters["n_neighbors"]}
            row = {"model": name, "family": family, "feature_group": group, "fold": number,
                   "features": len(features), "train_windows": len(train), "validation_windows": len(validation),
                   "validation_people": len(people), "eligible_people": int(people.two_class_defined.sum()),
                   "equal_person_balanced_accuracy": float(people.balanced_accuracy.mean()),
                   "equal_person_macro_f1": float(people.macro_f1.mean()),
                   "pooled_balanced_accuracy": metrics(pred.target, pred.predicted)["balanced_accuracy"],
                   "pooled_macro_f1": metrics(pred.target, pred.predicted)["macro_f1"],
                   "tuning_seconds": search_seconds, "selected_parameters": json.dumps(parameters, sort_keys=True),
                   "fit_median_seconds": float(np.median(fit_times)),
                   "fit_p25_seconds": float(np.quantile(fit_times, .25)), "fit_p75_seconds": float(np.quantile(fit_times, .75)),
                   "predict_median_seconds": float(np.median(predict_times)),
                   "predict_p25_seconds": float(np.quantile(predict_times, .25)), "predict_p75_seconds": float(np.quantile(predict_times, .75)),
                   "complexity": json.dumps(complexity, sort_keys=True)}
            fold_rows.append(row)
            print(f"Outer fold {number}: {name}: person BA {row['equal_person_balanced_accuracy']:.4f}; selected {parameters}", flush=True)
    folds = pd.DataFrame(fold_rows)
    people = pd.concat(person_rows, ignore_index=True)
    pred = pd.concat(predictions, ignore_index=True)
    assert not set(pred.pid) & locked
    assert not pred.duplicated(["model", "pid", "segment", "window_index"]).any()
    assert not people.duplicated(["model", "pid"]).any()
    assert len(pred) == len(data)*len(definitions)
    summary = []
    for name, own in people.groupby("model", sort=False):
        fold = folds[folds.model == name]
        pooled = pred[pred.model == name]
        summary.append({"model": name, "people": len(own), "eligible_people": int(own.two_class_defined.sum()),
                        "equal_person_balanced_accuracy": float(own.balanced_accuracy.mean()),
                        "person_ba_sd": float(own.balanced_accuracy.std(ddof=1)),
                        "equal_person_macro_f1": float(own.macro_f1.mean()),
                        "outer_fold_ba_sd": float(fold.equal_person_balanced_accuracy.std(ddof=1)),
                        "fold_ba_min": float(fold.equal_person_balanced_accuracy.min()),
                        "fold_ba_max": float(fold.equal_person_balanced_accuracy.max()),
                        "tuning_total_seconds": float(fold.tuning_seconds.sum()),
                        "refit_median_seconds_across_folds": float(fold.fit_median_seconds.median()),
                        "predict_median_seconds_across_folds": float(fold.predict_median_seconds.median()),
                        "pooled": metrics(pooled.target, pooled.predicted)})
    folds.to_csv(output / "fold_metrics.csv", index=False)
    people.to_csv(output / "participant_metrics.csv", index=False)
    pred.to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(searches).to_csv(output / "search.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(output / "timing.csv", index=False)
    pd.DataFrame([{k:v for k,v in row.items() if k != "pooled"} for row in summary]).to_csv(output / "summary.csv", index=False)
    write_json(output / "summary.json", summary)
    write_json(output / "inner_folds.json", inner_folds)
    write_json(output / "plan.json", {**plan, "feature_lists": groups, "outer_fold_manifest": outer_folds})
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    environment = {"command": ".venv/bin/python -B assignment3/src/compare.py", "python": sys.version,
                   "platform": platform.platform(), "cpu": platform.processor(), "logical_cpus": os.cpu_count(),
                   "packages": packages, "threads": {k:os.environ.get(k) for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]},
                   "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
                   "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT).decode(),
                   "source_sha256": sha256(__file__), "plan_sha256": sha256(plan_path),
                   "v2_data_sha256": sha256(dataset), "quality_selection_sha256": sha256(A3 / "results/quality/selection.json"),
                   "timing_method": "5 warm batches per operation; >=20ms timed work or 1000 calls; fresh clone outside each fit timer; scaler included; I/O, cloning and metrics excluded",
                   "full_development_final_fit": "not performed; selected outer refits only", "test_predictions": "not performed"}
    write_json(output / "environment.json", environment)
    write_json(output / "checks.json", {"status": "PASS", "development_windows": len(data), "people": int(data.pid.nunique()),
                                        "predictions": len(pred), "search_candidates": len(searches),
                                        "checks": ["V1 and A1 provenance", "V2 policy and hash", "disjoint nested person folds",
                                                   "no test predictions", "explicit input allowlists", "training-only scaler statistics",
                                                   "one-class two-class-metric suppression", "unique OOF predictions"]})
    print(pd.read_csv(output / "summary.csv").to_string(index=False))


if __name__ == "__main__":
    main()
