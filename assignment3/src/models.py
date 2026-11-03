"""V3 nested participant validation: basic models with feature selection, and
boosting/bagging ensembles. Never evaluate the locked final test."""
import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from functools import partial
from time import perf_counter

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from data import A3, ROOT, POSITION, sha256, write_json
from compare import metrics, participant_metrics, inner_partitions, repeated_time
from fixations import FEATURE_GROUPS_V3, KEY, fingerprint

PLAN = A3 / "model_plan_v3.json"
PARTS = {"basic": ["tree", "knn", "svm_rbf"], "ensembles": ["adaboost", "bagged_knn"]}
SELECTED = ("tree", "knn", "svm_rbf")


class BalancedAdaBoost(AdaBoostClassifier):
    """AdaBoost starting from class-balanced window weights. Balancing the base tree
    instead makes it optimise a different distribution than the one AdaBoost scores
    it on, so the first stump can exceed 0.5 weighted error (see plan amendment)."""

    def fit(self, X, y, sample_weight=None):
        assert sample_weight is None
        return super().fit(X, y, sample_weight=compute_sample_weight("balanced", y))


def grid(plan, family):
    """Basic families are tuned jointly with the filter size k."""
    values = dict(plan["grids"][family])
    if family in plan["feature_selection"]["applies_to"]:
        values["k"] = plan["feature_selection"]["k"]
    return list(ParameterGrid(values))


def estimator(family, parameters, seed, plan):
    parameters = dict(parameters)
    if family in SELECTED:
        mi = plan["feature_selection"]["mutual_info"]
        select = SelectKBest(partial(mutual_info_classif, n_neighbors=mi["n_neighbors"],
                                     random_state=mi["random_state"]), k=parameters.pop("k"))
        if family == "tree":
            return Pipeline([("select", select), ("model", DecisionTreeClassifier(
                **parameters, class_weight="balanced", random_state=seed))])
        model = (KNeighborsClassifier(**parameters, p=2, n_jobs=1) if family == "knn" else
                 SVC(**parameters, kernel="rbf", class_weight="balanced", cache_size=128))
        return Pipeline([("scale", StandardScaler()), ("select", select), ("model", model)])
    if family == "adaboost":
        base = DecisionTreeClassifier(random_state=seed)
        return BalancedAdaBoost(estimator=base, n_estimators=200, random_state=seed).set_params(**parameters)
    if family == "bagged_knn":
        base = make_pipeline(StandardScaler(), KNeighborsClassifier(p=2, n_jobs=1))
        return BaggingClassifier(estimator=base, n_estimators=50, bootstrap=True, bootstrap_features=False,
                                 random_state=seed, n_jobs=1).set_params(**parameters)
    if family == "dummy":
        return DummyClassifier(strategy="most_frequent")
    if family == "position_stump":
        return DecisionTreeClassifier(max_depth=1, min_samples_leaf=10, class_weight="balanced", random_state=seed)
    raise ValueError(family)


def fit_model(family, parameters, seed, train, features, plan):
    model = estimator(family, parameters, seed, plan)
    X, y = train[features].to_numpy(dtype=float), train.target.to_numpy()
    model.fit(X, y)
    if family in ("knn", "svm_rbf"):
        scaler = model.named_steps["scale"]
        assert scaler.n_samples_seen_ == len(train) and np.allclose(scaler.mean_, X.mean(axis=0))
    if family in SELECTED:
        assert model.named_steps["select"].n_features_in_ == len(features)
    if family == "bagged_knn":
        assert all(member.steps[0][1].n_samples_seen_ == len(train) for member in model.estimators_)
    return model


def tune(train, partitions, family, features, plan):
    results = []
    started = perf_counter()
    for parameters in grid(plan, family):
        predictions = []
        for part in partitions:
            assert not (set(part["train"]) & set(part["validation"]))
            inner_train = train[train.pid.isin(part["train"])]
            validation = train[train.pid.isin(part["validation"])]
            model = fit_model(family, parameters, plan["seed"], inner_train, features, plan)
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
    return best["parameters"], results, perf_counter() - started


def complexity(family, model, features, train, parameters):
    if family in SELECTED:
        kept = [f for f, keep in zip(features, model.named_steps["select"].get_support()) if keep]
        core = model.named_steps["model"]
        info = {"selected_features": kept}
        if family == "tree":
            info.update({"depth": int(core.get_depth()), "nodes": int(core.tree_.node_count)})
        if family == "knn":
            info.update({"stored_training_windows": len(train), "k": parameters["n_neighbors"]})
        if family == "svm_rbf":
            info["support_vectors"] = int(core.n_support_.sum())
        return info
    if family == "adaboost":
        return {"rounds": len(model.estimators_), "total_nodes": int(sum(t.tree_.node_count for t in model.estimators_))}
    if family == "bagged_knn":
        return {"members": len(model.estimators_), "stored_windows_per_member": len(train),
                "features_per_member": len(model.estimators_features_[0])}
    if family == "position_stump":
        axis = int(model.tree_.feature[0])
        return {"split_feature": features[axis] if axis >= 0 else None,
                "threshold_px": float(model.tree_.threshold[0]) if axis >= 0 else None}
    return {}


def verify(pred, people, folds, searches, families):
    """Independent recomputation, in the style of V2 verification.json."""
    for (model, pid), rows in pred.groupby(["model", "pid"]):
        own = people[(people.model == model) & (people.pid == pid)].iloc[0]
        tn, fp, fn, tp = confusion_matrix(rows.target, rows.predicted, labels=[0, 1]).ravel()
        assert (tn, fp, fn, tp) == (own.tn, own.fp, own.fn, own.tp)
        if rows.target.nunique() == 2:
            assert np.isclose(balanced_accuracy_score(rows.target, rows.predicted), own.balanced_accuracy)
    search = pd.DataFrame(searches)
    for row in folds[folds.family.isin(families)].itertuples():
        own = search[(search.model == row.model) & (search.outer_fold == row.fold)]
        best = own.loc[own.inner_equal_person_balanced_accuracy.idxmax(), "parameters"]
        assert best == row.selected_parameters
    return {"status": "PASS", "prediction_rows": len(pred), "participant_metric_rows": len(people),
            "independent_sklearn_metrics_and_confusions": "PASS",
            "selected_parameters_match_inner_search_only": "PASS", "locked_test_predictions": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", choices=sorted(PARTS))
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    selection = json.loads((A3 / "results/quality/selection.json").read_text())
    locked = set(selection["policy"]["test_participants"])
    full = pd.read_parquet(ROOT / plan["dataset"])
    assert fingerprint(full, FEATURE_GROUPS_V3["A3_with_position"] + ["target", "split"]) == plan["dataset_content_sha256"]
    data = full[full.split == "development"].reset_index(drop=True)
    del full
    assert not (set(data.pid) & locked)
    outer_folds = json.loads((ROOT / plan["outer_folds"]).read_text())
    validation_people = [pid for fold in outer_folds for pid in fold["validation_participants"]]
    assert len(validation_people) == len(set(validation_people)) and set(validation_people) == set(data.pid)
    groups = {**FEATURE_GROUPS_V3, "position_only": POSITION}
    forbidden = {"pid", "target", "stimulus", "segment", "window_index", "start_us", "end_us", "split",
                 "measured_fraction", "accepted_v2", "language", "file", "fixation_count", "saccade_count"}
    for names in groups.values():
        assert not (set(names) & forbidden) and np.isfinite(data[names]).all().all()
    assert not set(POSITION) & set(groups["B3_without_absolute_position"])
    families = PARTS[args.part]
    definitions = [(family, group) for family in families for group in plan["feature_groups"]]
    if args.part == "basic":
        definitions += [("dummy", "B3_without_absolute_position"), ("position_stump", "position_only")]
    output = A3 / "results/models_v3" / args.part
    output.mkdir(parents=True, exist_ok=True)
    fold_rows, person_rows, predictions, searches, timing_rows, importance_rows = [], [], [], [], [], []
    for fold in outer_folds:
        number = fold["fold"]
        tr_ids, va_ids = set(fold["train_participants"]), set(fold["validation_participants"])
        assert not tr_ids & va_ids and not (tr_ids | va_ids) & locked and tr_ids | va_ids == set(data.pid)
        train, validation = data[data.pid.isin(tr_ids)], data[data.pid.isin(va_ids)]
        partitions = inner_partitions(tr_ids, plan["inner_group_folds"], plan["seed"])
        for part in partitions:
            assert set(part["train"]) | set(part["validation"]) == tr_ids
        for family, group in definitions:
            name = f"{family}__{group}"
            features = groups[group]
            parameters, search_seconds = {}, 0.0
            if family in plan["grids"]:
                parameters, candidates, search_seconds = tune(train, partitions, family, features, plan)
                for candidate in candidates:
                    searches.append({"model": name, "outer_fold": number,
                                     "parameters": json.dumps(candidate["parameters"], sort_keys=True),
                                     "inner_equal_person_balanced_accuracy": candidate["inner_equal_person_balanced_accuracy"],
                                     "eligible_people": candidate["eligible_people"]})
            model = fit_model(family, parameters, plan["seed"], train, features, plan)
            Xtr, Xval = train[features].to_numpy(dtype=float), validation[features].to_numpy(dtype=float)
            pred = validation[KEY + ["target"]].copy()
            pred["predicted"] = model.predict(Xval)
            pred["model"], pred["fold"] = name, number
            predictions.append(pred)
            people = participant_metrics(pred)
            people["model"], people["fold"] = name, number
            person_rows.append(people)
            times = repeated_time(model, Xtr, train.target.to_numpy(), Xval, plan["timing"])
            for row in times:
                timing_rows.append({"model": name, "fold": number, **row})
            fit_times = [r["seconds_per_call"] for r in times if r["operation"] == "fit"]
            predict_times = [r["seconds_per_call"] for r in times if r["operation"] == "predict"]
            if family in plan["grids"]:
                importance = permutation_importance(model, Xval, validation.target.to_numpy(), scoring="balanced_accuracy",
                                                    n_repeats=10, random_state=plan["seed"], n_jobs=1)
                for feature, mean, sd in zip(features, importance.importances_mean, importance.importances_std):
                    importance_rows.append({"model": name, "fold": number, "feature": feature,
                                            "importance_mean": float(mean), "importance_sd": float(sd)})
            pooled = metrics(pred.target, pred.predicted)
            fold_rows.append({"model": name, "family": family, "feature_group": group, "fold": number,
                              "features": len(features), "train_windows": len(train), "validation_windows": len(validation),
                              "validation_people": len(people), "eligible_people": int(people.two_class_defined.sum()),
                              "equal_person_balanced_accuracy": float(people.balanced_accuracy.mean()),
                              "equal_person_macro_f1": float(people.macro_f1.mean()),
                              "pooled_balanced_accuracy": pooled["balanced_accuracy"], "pooled_macro_f1": pooled["macro_f1"],
                              "tuning_seconds": search_seconds, "selected_parameters": json.dumps(parameters, sort_keys=True),
                              "fit_median_seconds": float(np.median(fit_times)),
                              "fit_p25_seconds": float(np.quantile(fit_times, .25)), "fit_p75_seconds": float(np.quantile(fit_times, .75)),
                              "predict_median_seconds": float(np.median(predict_times)),
                              "predict_p25_seconds": float(np.quantile(predict_times, .25)),
                              "predict_p75_seconds": float(np.quantile(predict_times, .75)),
                              "complexity": json.dumps(complexity(family, model, features, train, parameters), sort_keys=True)})
            print(f"Outer fold {number}: {name}: person BA {fold_rows[-1]['equal_person_balanced_accuracy']:.4f}; "
                  f"selected {parameters}", flush=True)
    folds = pd.DataFrame(fold_rows)
    people = pd.concat(person_rows, ignore_index=True)
    pred = pd.concat(predictions, ignore_index=True)
    assert not set(pred.pid) & locked
    assert not pred.duplicated(["model"] + KEY).any() and not people.duplicated(["model", "pid"]).any()
    assert len(pred) == len(data) * len(definitions)
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
    verification = verify(pred, people, folds, searches, [f for f in families if f in plan["grids"]])
    folds.to_csv(output / "fold_metrics.csv", index=False)
    people.to_csv(output / "participant_metrics.csv", index=False)
    pred.to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(searches).to_csv(output / "search.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(output / "timing.csv", index=False)
    pd.DataFrame(importance_rows).to_csv(output / "permutation_importance.csv", index=False)
    pd.DataFrame([{k: v for k, v in row.items() if k != "pooled"} for row in summary]).to_csv(output / "summary.csv", index=False)
    write_json(output / "summary.json", summary)
    write_json(output / "verification.json", verification)
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    write_json(output / "environment.json", {
        "command": f".venv/bin/python -B assignment3/src/models.py {args.part}", "python": sys.version,
        "platform": platform.platform(), "cpu": platform.processor(), "logical_cpus": os.cpu_count(),
        "packages": packages, "threads": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT).decode(),
        "source_sha256": sha256(__file__), "plan_sha256": sha256(PLAN),
        "v3_content_sha256": plan["dataset_content_sha256"],
        "timing_method": "as V2: 5 warm batches per operation; >=20ms timed work or 1000 calls; fresh clone outside each fit timer",
        "full_development_final_fit": "not performed; selected outer refits only", "test_predictions": "not performed"})
    write_json(output / "checks.json", {"status": "PASS", "development_windows": len(data), "people": int(data.pid.nunique()),
                                        "predictions": len(pred), "search_candidates": len(searches),
                                        "checks": ["V3 content fingerprint", "disjoint nested person folds", "no test predictions",
                                                   "explicit input allowlists", "B3 has no absolute position",
                                                   "training-only scaler and selector", "unique OOF predictions",
                                                   "independent sklearn recomputation"]})
    print(pd.read_csv(output / "summary.csv")[["model", "equal_person_balanced_accuracy", "person_ba_sd",
                                              "fold_ba_min", "fold_ba_max"]].to_string(index=False))


if __name__ == "__main__":
    main()
