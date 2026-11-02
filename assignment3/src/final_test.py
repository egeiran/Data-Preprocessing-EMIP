"""One-time locked-test evaluation of every declared method (final_test_plan.json).
Refuses to run twice. No score is printed until every model is fitted and saved."""
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
import torch
from sklearn.model_selection import ParameterGrid

from data import A3, ROOT, FEATURE_GROUPS, POSITION, sha256, write_json
import compare
import models
import cnn
from compare import metrics, participant_metrics, inner_partitions, repeated_time
from fixations import FEATURE_GROUPS_V3, KEY, fingerprint
from sequences import digest

PLAN = A3 / "final_test_plan.json"
OUTPUT = A3 / "results/final_test"
MODE_LR = {"freeze": 0.001, "finetune": 0.0003}  # as cnn.py and deep_plan.json


def cnn_oof(meta, raw, y, partitions, group, config, deep, init_for=None, mode=None):
    """As cnn.py: pooled out-of-fold predictions at each partition's best epoch."""
    frames, epochs = [], []
    for index, part in enumerate(partitions):
        tr, va = meta.pid.isin(part["train"]).to_numpy(), meta.pid.isin(part["validation"]).to_numpy()
        pupil = cnn.pupil_stats(raw[tr])
        init = init_for(index, part, pupil) if init_for else None
        _, best = cnn.fit(cnn.prepare(raw[tr], group, pupil), y[tr], config, deep["fixed"]["max_epochs"], deep["seed"],
                          deep["fixed"], validation=(cnn.prepare(raw[va], group, pupil), meta[va]), init=init, mode=mode,
                          lr=MODE_LR.get(mode))
        frame = meta.loc[va, ["pid", "target"]].copy()
        frame["predicted"] = best["predicted"]
        frames.append(frame)
        epochs.append(best["epoch"])
    oof = pd.concat(frames)
    assert len(oof) == len(meta) and not oof.index.duplicated().any()
    return float(participant_metrics(oof).balanced_accuracy.mean()), epochs


def interval(values, resamples):
    means = values[resamples].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main():
    if OUTPUT.exists():
        raise SystemExit(f"{OUTPUT} exists: the locked test has already been evaluated once.")
    plan = json.loads(PLAN.read_text())
    v2_plan = json.loads((A3 / "comparison_plan.json").read_text())
    v3_plan = json.loads((A3 / "model_plan_v3.json").read_text())
    deep = json.loads((A3 / "deep_plan.json").read_text())
    locked = set(plan["locked_test_participants"])
    full = pd.read_parquet(ROOT / v3_plan["dataset"])
    assert fingerprint(full, FEATURE_GROUPS_V3["A3_with_position"] + ["target", "split"]) == v3_plan["dataset_content_sha256"]
    development = full[full.split == "development"].reset_index(drop=True)
    test = full[full.split == "locked_test"].reset_index(drop=True)
    assert set(test.pid) == locked and not set(development.pid) & locked and development.pid.nunique() == 23
    dev_ids = set(development.pid)
    partitions = inner_partitions(dev_ids, 3, plan["seed"])
    for part in partitions:
        assert set(part["train"]) | set(part["validation"]) == dev_ids and not set(part["train"]) & set(part["validation"])
    groups = {**FEATURE_GROUPS, **FEATURE_GROUPS_V3, "position_only": POSITION}
    fitted, selection_rows, timing_rows = [], [], []
    # Feature-based methods: select on development partitions, refit on all development windows.
    definitions = [("v2_basic", f, g) for f in plan["methods"]["v2_basic"]["families"]
                   for g in plan["methods"]["v2_basic"]["feature_groups"]]
    definitions += [(m, f, g) for m in ("v3_basic_with_selection", "v3_ensembles")
                    for f in plan["methods"][m]["families"] for g in plan["methods"][m]["feature_groups"]]
    definitions += [("control", "dummy", "B3_without_absolute_position"), ("control", "position_stump", "position_only")]
    for method, family, group in definitions:
        features = groups[group]
        started = perf_counter()
        if method == "v2_basic":
            parameters, candidates, _ = compare.tune(development, partitions, family, features, v2_plan["grids"][family], plan["seed"])
            model = compare.fit_model(family, parameters, plan["seed"], development, features)
        elif method == "control":
            parameters, candidates = {}, []
            model = models.fit_model(family, parameters, plan["seed"], development, features, v3_plan)
        else:
            parameters, candidates, _ = models.tune(development, partitions, family, features, v3_plan)
            model = models.fit_model(family, parameters, plan["seed"], development, features, v3_plan)
        selection_seconds = perf_counter() - started
        for candidate in candidates:
            selection_rows.append({"method": method, "model": f"{family}__{group}",
                                   "parameters": json.dumps(candidate["parameters"], sort_keys=True),
                                   "development_cv_equal_person_balanced_accuracy": candidate["inner_equal_person_balanced_accuracy"]})
        Xdev, Xtest = development[features].to_numpy(dtype=float), test[features].to_numpy(dtype=float)
        times = repeated_time(model, Xdev, development.target.to_numpy(), Xtest, v3_plan["timing"])
        timing_rows += [{"method": method, "model": f"{family}__{group}", **row} for row in times]
        info = models.complexity(family, model, features, development, parameters) if method != "v2_basic" else {}
        fitted.append({"method": method, "family": family, "feature_group": group, "model": f"{family}__{group}",
                       "predicted": model.predict(Xtest), "parameters": parameters, "selection_seconds": selection_seconds,
                       "fit_seconds": float(np.median([r["seconds_per_call"] for r in times if r["operation"] == "fit"])),
                       "predict_seconds": float(np.median([r["seconds_per_call"] for r in times if r["operation"] == "predict"])),
                       "complexity": info, "seed_scores": {}})
        print(f"Fitted {method} {family}__{group}", flush=True)
    # Sequence methods.
    torch.set_num_threads(cnn.TRAIN_THREADS)
    torch.use_deterministic_algorithms(True)
    evidence = json.loads((A3 / "results/deep_v3/sequences.json").read_text())
    npz = np.load(A3 / "cache/sequences.npz")
    assert digest(npz["code_X"], npz["code_target"]) == evidence["code_sha256"]
    meta_all = pd.DataFrame({"pid": npz["code_pid"], "segment": npz["code_segment"],
                             "window_index": npz["code_window_index"], "target": npz["code_target"], "split": npz["code_split"]})
    dev_mask, test_mask = (meta_all.split == "development").to_numpy(), (meta_all.split == "locked_test").to_numpy()
    meta, raw, y = meta_all[dev_mask].reset_index(drop=True), npz["code_X"][dev_mask], npz["code_target"][dev_mask]
    test_meta, test_raw = meta_all[test_mask].reset_index(drop=True), npz["code_X"][test_mask]
    assert test_meta[KEY].sort_values(KEY).reset_index(drop=True).equals(test[KEY].sort_values(KEY).reset_index(drop=True))
    source_raw, source_pid, source_y = npz["source_X"], npz["source_pid"], npz["source_target"]
    assert not set(source_pid) & locked
    pupil = cnn.pupil_stats(raw)
    seeds = [deep["seeds"]["primary"]] + deep["seeds"]["spread"]
    for group in cnn.GROUPS:
        Xdev, Xtest = cnn.prepare(raw, group, pupil), cnn.prepare(test_raw, group, pupil)
        started = perf_counter()
        results = []
        for config in ParameterGrid(deep["grid"]):
            score, epochs = cnn_oof(meta, raw, y, partitions, group, config, deep)
            results.append({"config": config, "score": score, "epochs": epochs})
            selection_rows.append({"method": "deep", "model": f"cnn__{group}", "parameters": json.dumps(config, sort_keys=True),
                                   "development_cv_equal_person_balanced_accuracy": score})
        chosen = max(results, key=lambda r: r["score"])
        config, scratch_seconds = chosen["config"], perf_counter() - started
        started = perf_counter()
        cache = {}

        def source_for(people, part_pupil):
            mask = np.isin(source_pid, sorted(people))
            assert set(source_pid[mask]) <= set(people)
            return cnn.prepare(source_raw[mask], group, part_pupil), source_y[mask]

        def init_for(index, part, part_pupil):
            if index not in cache:
                cache[index] = cnn.pretrain(*source_for(part["train"], part_pupil), config, deep)
            return cache[index]

        modes = []
        for mode in deep["transfer"]["fine_tune_grid"]["mode"]:
            score, epochs = cnn_oof(meta, raw, y, partitions, group, config, deep, init_for, mode)
            modes.append({"mode": mode, "score": score, "epochs": epochs})
            selection_rows.append({"method": "deep", "model": f"transfer__{group}", "parameters": json.dumps({"mode": mode}),
                                   "development_cv_equal_person_balanced_accuracy": score})
        chosen_mode = max(modes, key=lambda r: r["score"])
        final_init = cnn.pretrain(*source_for(dev_ids, pupil), config, deep)
        transfer_seconds = perf_counter() - started
        variants = [("cnn", None, None, int(np.median(chosen["epochs"])), dict(config), scratch_seconds),
                    ("transfer", final_init, chosen_mode["mode"], int(np.median(chosen_mode["epochs"])),
                     {**config, "mode": chosen_mode["mode"]}, transfer_seconds)]
        for family, init, mode, epochs, parameters, search_seconds in variants:
            seed_scores, primary = {}, None
            for seed in seeds:
                started = perf_counter()
                net, _ = cnn.fit(Xdev, y, config, epochs, seed, deep["fixed"], init=init, mode=mode, lr=MODE_LR.get(mode))
                fit_seconds = perf_counter() - started
                predicted = cnn.predict(net, Xtest)
                if seed == deep["seeds"]["primary"]:
                    times = cnn.timed_predict(net, Xtest)
                    timing_rows += [{"method": "deep", "model": f"{family}__{group}", **row} for row in times]
                    primary = (predicted, fit_seconds, float(np.median([r["seconds_per_call"] for r in times])),
                               {"epochs": epochs, "parameters_total": sum(p.numel() for p in net.parameters()),
                                "parameters_trainable": sum(p.numel() for p in net.parameters() if p.requires_grad)})
                else:
                    seed_scores[seed] = predicted
            fitted.append({"method": "deep", "family": family, "feature_group": group, "model": f"{family}__{group}",
                           "predicted": primary[0], "parameters": parameters, "selection_seconds": search_seconds,
                           "fit_seconds": primary[1], "predict_seconds": primary[2], "complexity": primary[3],
                           "seed_scores": seed_scores, "test_meta": test_meta})
            print(f"Fitted deep {family}__{group}", flush=True)
    # Score every model once.
    resamples = np.random.default_rng(plan["seed"]).integers(0, 6, size=(10_000, 6))
    predictions, person_rows, summary, seed_rows = [], [], [], []
    for item in fitted:
        frame = item.get("test_meta", test)[KEY + ["target"]].copy()
        frame["predicted"], frame["model"], frame["method"] = item["predicted"], item["model"], item["method"]
        predictions.append(frame)
        people = participant_metrics(frame)
        people["model"], people["method"] = item["model"], item["method"]
        person_rows.append(people)
        eligible = people[people.two_class_defined].sort_values("pid", key=lambda s: s.astype(int))
        assert len(eligible) == 6 and set(people.pid) - set(eligible.pid) == {"18"}
        for seed, predicted in item["seed_scores"].items():
            other = frame.assign(predicted=predicted)
            seed_rows.append({"model": item["model"], "seed": seed,
                              "equal_person_balanced_accuracy": float(participant_metrics(other).balanced_accuracy.mean())})
        summary.append({"method": item["method"], "model": item["model"], "family": item["family"],
                        "feature_group": item["feature_group"],
                        "test_equal_person_balanced_accuracy": float(eligible.balanced_accuracy.mean()),
                        "test_ba_interval_95": interval(eligible.balanced_accuracy.to_numpy(), resamples),
                        "test_equal_person_macro_f1": float(eligible.macro_f1.mean()),
                        "test_person_ba_min": float(eligible.balanced_accuracy.min()),
                        "test_person_ba_max": float(eligible.balanced_accuracy.max()),
                        "pid18_vehicle_recall": float(people.loc[people.pid == "18", "observed_class_recall"].iloc[0]),
                        "pooled": metrics(frame.target, frame.predicted),
                        "selected_parameters": json.dumps(item["parameters"], sort_keys=True),
                        "selection_seconds": item["selection_seconds"], "refit_seconds": item["fit_seconds"],
                        "predict_seconds_test_windows": item["predict_seconds"], "complexity": json.dumps(item["complexity"], sort_keys=True)})
    people = pd.concat(person_rows, ignore_index=True)
    differences = []
    for (method, family), own in pd.DataFrame(summary).groupby(["method", "family"], sort=False):
        if len(own) != 2:
            continue
        a, b = sorted(own.model)
        rows = {m: people[(people.model == m) & people.two_class_defined].set_index("pid").balanced_accuracy for m in (a, b)}
        diff = (rows[a] - rows[b]).loc[sorted(rows[a].index, key=int)].to_numpy()
        differences.append({"method": method, "family": family, "with_position": a, "without_position": b,
                            "mean_ba_difference": float(diff.mean()), "interval_95": interval(diff, resamples),
                            "people_with_position_better": int((diff > 0).sum())})
    OUTPUT.mkdir(parents=True)
    pd.concat(predictions, ignore_index=True).to_csv(OUTPUT / "predictions.csv", index=False)
    people.to_csv(OUTPUT / "participant_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(OUTPUT / "selection.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(OUTPUT / "timing.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(OUTPUT / "seeds.csv", index=False)
    pd.DataFrame([{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in row.items() if k != "pooled"}
                  for row in summary]).to_csv(OUTPUT / "summary.csv", index=False)
    write_json(OUTPUT / "summary.json", summary)
    write_json(OUTPUT / "differences.json", differences)
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    write_json(OUTPUT / "environment.json", {
        "command": ".venv/bin/python -B assignment3/src/final_test.py", "python": sys.version,
        "platform": platform.platform(), "cpu": platform.processor(), "logical_cpus": os.cpu_count(), "packages": packages,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT).decode(),
        "source_sha256": sha256(__file__), "plan_sha256": sha256(PLAN), "v3_content_sha256": v3_plan["dataset_content_sha256"],
        "sequences": evidence["code_sha256"], "runs": 1})
    write_json(OUTPUT / "checks.json", {"status": "PASS", "test_windows": len(test), "test_people": len(locked),
                                        "eligible_two_class_people": 6, "models": len(fitted),
                                        "checks": ["V3 content fingerprint and sequence digest", "selection on development people only",
                                                   "pretraining on development people only", "test keys match between tables",
                                                   "pid 18 excluded from two-class metrics", "single run guard"]})
    columns = ["model", "method", "test_equal_person_balanced_accuracy", "test_ba_interval_95"]
    print(pd.DataFrame(summary)[columns].to_string(index=False))
    print(json.dumps(differences, indent=2))


if __name__ == "__main__":
    main()
