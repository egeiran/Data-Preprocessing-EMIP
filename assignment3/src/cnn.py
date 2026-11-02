"""Sequence CNN with nested participant tuning and screen-type transfer.
Never evaluate the locked final test."""
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import ParameterGrid
from torch import nn
import torch.nn.functional as F

from data import A3, ROOT, sha256, write_json
from compare import metrics, participant_metrics, inner_partitions
from sequences import digest

PLAN = A3 / "deep_plan.json"
GROUPS = ["A_seq_with_position", "B_seq_without_absolute_position"]
SCREEN = (1920.0, 1080.0)
TRAIN_THREADS = 4
PRETRAIN_EPOCHS = 30
KEY = ["pid", "segment", "window_index"]


class Conv1d(nn.Conv1d):
    """Same parameters and initialisation as nn.Conv1d with odd kernel and same padding,
    computed as unfold + einsum. torch on macOS arm64 has no oneDNN and falls back to a
    convolution ~25x slower; outputs agree to float rounding (WORKLOG)."""

    def forward(self, x):
        pad = self.kernel_size[0] // 2
        columns = F.pad(x, (pad, pad)).unfold(2, self.kernel_size[0], 1)
        return torch.einsum("bclk,ock->bol", columns, self.weight) + self.bias[None, :, None]


class Net(nn.Module):
    def __init__(self, filters, kernel_size, channels=6):
        super().__init__()

        def block(inputs, outputs):
            return [Conv1d(inputs, outputs, kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(outputs), nn.ReLU()]

        self.features = nn.Sequential(*block(channels, filters), nn.MaxPool1d(2),
                                      *block(filters, 2 * filters), nn.MaxPool1d(2),
                                      *block(2 * filters, 2 * filters), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * filters, 2))

    def forward(self, x):
        return self.head(self.features(x))


def pupil_stats(raw):
    """Mean/SD of present pupil samples; always called on training windows only."""
    present = (raw[:, 4] + raw[:, 5]) > 0
    return [(float(raw[:, c][present].mean()), float(raw[:, c][present].std())) for c in (2, 3)]


def prepare(raw, group, pupil):
    X = raw.astype(np.float64)
    present = (X[:, 4] + X[:, 5]) > 0
    if group.startswith("B"):
        # Remove each window's mean position: relative movement stays, location goes.
        for c in (0, 1):
            mean = (X[:, c] * present).sum(axis=1) / present.sum(axis=1)
            X[:, c] = np.where(present, X[:, c] - mean[:, None], 0)
            assert np.abs((X[:, c] * present).sum(axis=1)).max() < 1e-6 * present.sum(axis=1).max()
    X[:, 0] /= SCREEN[0]
    X[:, 1] /= SCREEN[1]
    for c, (mu, sd) in zip((2, 3), pupil):
        X[:, c] = np.where(present, (X[:, c] - mu) / sd, 0)
    return X.astype(np.float32)


def class_weights(y):
    counts = np.bincount(y, minlength=2)
    return torch.tensor(len(y) / (2 * counts), dtype=torch.float32)


def predict(net, X, batch=256):
    net.eval()
    with torch.no_grad():
        return np.concatenate([net(torch.from_numpy(X[i:i + batch])).argmax(1).numpy()
                               for i in range(0, len(X), batch)])


def person_ba(meta, predicted):
    frame = meta[["pid", "target"]].copy()
    frame["predicted"] = predicted
    return participant_metrics(frame).balanced_accuracy.mean()


def fit(X, y, config, epochs, seed, fixed, validation=None, init=None, mode=None, lr=None):
    """Train for `epochs`; with validation (X, meta) keep the first best epoch and stop
    after `patience` epochs without improvement."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net(config["filters"], config["kernel_size"])
    if init is not None:
        net.features.load_state_dict(init)
    if mode == "freeze":
        for p in net.features.parameters():
            p.requires_grad_(False)
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr or fixed["learning_rate"])
    loss = nn.CrossEntropyLoss(weight=class_weights(y))
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.int64))
    best = {"score": -np.inf, "epoch": 0, "predicted": None}
    for epoch in range(1, epochs + 1):
        net.train()
        if mode == "freeze":
            net.features.eval()  # frozen batch-norm statistics stay as pretrained
        order = rng.permutation(len(Xt))
        for start in range(0, len(order), fixed["batch_size"]):
            chunk = torch.from_numpy(order[start:start + fixed["batch_size"]])
            if len(chunk) < 2:  # batch norm needs two windows in training mode
                continue
            optimizer.zero_grad()
            loss(net(Xt[chunk]), yt[chunk]).backward()
            optimizer.step()
        if validation is not None:
            predicted = predict(net, validation[0])
            score = person_ba(validation[1], predicted)
            if score > best["score"]:
                best = {"score": float(score), "epoch": epoch, "predicted": predicted}
            elif epoch - best["epoch"] >= fixed["patience"]:
                break
    return net, best


def pretrain(source_X, source_y, config, plan):
    """Source task: instruction vs answer screen, fixed epochs (deep_plan transfer)."""
    net, _ = fit(source_X, source_y, config, PRETRAIN_EPOCHS, plan["seed"], plan["fixed"])
    return {k: v.clone() for k, v in net.features.state_dict().items()}


def timed_predict(net, X, batches=5, min_seconds=0.02, max_calls=1000):
    """Warm, single-threaded prediction batches, comparable to the sklearn timing."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    predict(net, X)
    rows = []
    for batch in range(batches):
        elapsed, calls = 0.0, 0
        while elapsed < min_seconds and calls < max_calls:
            started = perf_counter()
            predict(net, X)
            elapsed += perf_counter() - started
            calls += 1
        rows.append({"operation": "predict", "batch": batch + 1, "calls": calls,
                     "batch_seconds": elapsed, "seconds_per_call": elapsed / calls})
    torch.set_num_threads(threads)
    return rows


def main():
    plan = json.loads(PLAN.read_text())
    torch.set_num_threads(TRAIN_THREADS)
    torch.use_deterministic_algorithms(True)
    evidence = json.loads((A3 / "results/deep_v3/sequences.json").read_text())
    selection = json.loads((A3 / "results/quality/selection.json").read_text())
    locked = set(selection["policy"]["test_participants"])
    npz = np.load(A3 / "cache/sequences.npz")
    assert digest(npz["code_X"], npz["code_target"]) == evidence["code_sha256"]
    assert digest(npz["source_X"], npz["source_target"]) == evidence["source_sha256"]
    development = npz["code_split"] == "development"
    raw = npz["code_X"][development]
    meta = pd.DataFrame({"pid": npz["code_pid"], "segment": npz["code_segment"],
                         "window_index": npz["code_window_index"], "target": npz["code_target"]})[development].reset_index(drop=True)
    source_raw, source_pid, source_y = npz["source_X"], npz["source_pid"], npz["source_target"]
    assert not (set(meta.pid) | set(source_pid)) & locked
    outer_folds = json.loads((ROOT / plan["outer_folds"]).read_text())
    assert {p for f in outer_folds for p in f["validation_participants"]} == set(meta.pid)
    fixed, grid = plan["fixed"], list(ParameterGrid(plan["grid"]))
    modes = plan["transfer"]["fine_tune_grid"]["mode"]
    mode_lr = {"freeze": 0.001, "finetune": 0.0003}
    seeds = [plan["seeds"]["primary"]] + plan["seeds"]["spread"]
    y = meta.target.to_numpy()
    fold_rows, person_rows, predictions, searches, seed_rows, timing_rows = [], [], [], [], [], []

    def inner_oof(train_ids, partitions, group, config, init_for=None, mode=None):
        """Pooled inner out-of-fold predictions at each partition's best epoch."""
        frames, epochs = [], []
        for index, part in enumerate(partitions):
            tr, va = meta.pid.isin(part["train"]).to_numpy(), meta.pid.isin(part["validation"]).to_numpy()
            assert set(meta.pid[tr]) | set(meta.pid[va]) == train_ids and not set(meta.pid[tr]) & set(meta.pid[va])
            pupil = pupil_stats(raw[tr])
            init = init_for(index, part, pupil) if init_for else None
            _, best = fit(prepare(raw[tr], group, pupil), y[tr], config, fixed["max_epochs"], plan["seed"], fixed,
                          validation=(prepare(raw[va], group, pupil), meta[va]), init=init, mode=mode,
                          lr=mode_lr.get(mode))
            frame = meta.loc[va, ["pid", "target"]].copy()
            frame["predicted"] = best["predicted"]
            frames.append(frame)
            epochs.append(best["epoch"])
        oof = pd.concat(frames)
        assert len(oof) == sum(meta.pid.isin(train_ids)) and not oof.index.duplicated().any()
        return float(participant_metrics(oof).balanced_accuracy.mean()), epochs

    def source_for(people, group, pupil):
        mask = np.isin(source_pid, sorted(people))
        assert set(source_pid[mask]) <= set(people)
        return prepare(source_raw[mask], group, pupil), source_y[mask]

    for fold in outer_folds:
        number = fold["fold"]
        tr_ids, va_ids = set(fold["train_participants"]), set(fold["validation_participants"])
        assert not tr_ids & va_ids and not (tr_ids | va_ids) & locked
        tr, va = meta.pid.isin(tr_ids).to_numpy(), meta.pid.isin(va_ids).to_numpy()
        partitions = inner_partitions(tr_ids, plan["inner_group_folds"], plan["seed"])
        pupil = pupil_stats(raw[tr])
        for group in GROUPS:
            Xtr, Xva = prepare(raw[tr], group, pupil), prepare(raw[va], group, pupil)
            # Scratch CNN: tune filters x kernel on inner people only.
            started = perf_counter()
            results = []
            for config in grid:
                score, epochs = inner_oof(tr_ids, partitions, group, config)
                results.append({"config": config, "score": score, "epochs": epochs})
                searches.append({"model": f"cnn__{group}", "outer_fold": number, "parameters": json.dumps(config, sort_keys=True),
                                 "inner_equal_person_balanced_accuracy": score, "best_epochs": json.dumps(epochs)})
            chosen = max(results, key=lambda r: r["score"])
            scratch_search = perf_counter() - started
            config, refit_epochs = chosen["config"], int(np.median(chosen["epochs"]))
            # Transfer: pretrain per partition on its training people's other screens.
            started = perf_counter()
            cache = {}

            def init_for(index, part, part_pupil):
                if index not in cache:
                    cache[index] = pretrain(*source_for(part["train"], group, part_pupil), config, plan)
                return cache[index]

            mode_results = []
            for mode in modes:
                score, epochs = inner_oof(tr_ids, partitions, group, config, init_for, mode)
                mode_results.append({"mode": mode, "score": score, "epochs": epochs})
                searches.append({"model": f"transfer__{group}", "outer_fold": number,
                                 "parameters": json.dumps({"mode": mode}, sort_keys=True),
                                 "inner_equal_person_balanced_accuracy": score, "best_epochs": json.dumps(epochs)})
            chosen_mode = max(mode_results, key=lambda r: r["score"])
            transfer_search = perf_counter() - started
            started = perf_counter()
            outer_init = pretrain(*source_for(tr_ids, group, pupil), config, plan)
            pretrain_seconds = perf_counter() - started
            variants = [("cnn", None, None, refit_epochs, {**config}, scratch_search, 0.0),
                        ("transfer", outer_init, chosen_mode["mode"], int(np.median(chosen_mode["epochs"])),
                         {**config, "mode": chosen_mode["mode"]}, transfer_search, pretrain_seconds)]
            for family, init, mode, epochs, selected, search_seconds, pre_seconds in variants:
                name = f"{family}__{group}"
                for seed in seeds:
                    started = perf_counter()
                    net, _ = fit(Xtr, y[tr], config, epochs, seed, fixed, init=init, mode=mode, lr=mode_lr.get(mode))
                    fit_seconds = perf_counter() - started
                    predicted = predict(net, Xva)
                    seed_rows.append({"model": name, "fold": number, "seed": seed,
                                      "equal_person_balanced_accuracy": person_ba(meta[va], predicted), "fit_seconds": fit_seconds})
                    if seed != plan["seeds"]["primary"]:
                        continue
                    pred = meta.loc[va, KEY + ["target"]].copy()
                    pred["predicted"], pred["model"], pred["fold"] = predicted, name, number
                    predictions.append(pred)
                    people = participant_metrics(pred)
                    people["model"], people["fold"] = name, number
                    person_rows.append(people)
                    times = timed_predict(net, Xva)
                    timing_rows += [{"model": name, "fold": number, **row} for row in times]
                    pooled = metrics(pred.target, pred.predicted)
                    row = {"model": name, "family": family, "feature_group": group, "fold": number,
                           "train_windows": int(tr.sum()), "validation_windows": int(va.sum()),
                           "validation_people": len(people), "eligible_people": int(people.two_class_defined.sum()),
                           "equal_person_balanced_accuracy": float(people.balanced_accuracy.mean()),
                           "equal_person_macro_f1": float(people.macro_f1.mean()),
                           "pooled_balanced_accuracy": pooled["balanced_accuracy"], "pooled_macro_f1": pooled["macro_f1"],
                           "selected_parameters": json.dumps(selected, sort_keys=True), "refit_epochs": epochs,
                           "tuning_seconds": search_seconds, "pretrain_seconds": pre_seconds, "fit_seconds": fit_seconds,
                           "predict_median_seconds": float(np.median([r["seconds_per_call"] for r in times])),
                           "parameters_total": sum(p.numel() for p in net.parameters()),
                           "parameters_trainable": sum(p.numel() for p in net.parameters() if p.requires_grad)}
                    fold_rows.append(row)
                    print(f"Outer fold {number}: {name}: person BA {row['equal_person_balanced_accuracy']:.4f}; "
                          f"selected {selected}, {epochs} epochs", flush=True)
    folds, people = pd.DataFrame(fold_rows), pd.concat(person_rows, ignore_index=True)
    pred, seeds_frame = pd.concat(predictions, ignore_index=True), pd.DataFrame(seed_rows)
    assert not set(pred.pid) & locked and not pred.duplicated(["model"] + KEY).any()
    assert len(pred) == len(meta) * 2 * len(GROUPS)
    summary = []
    for name, own in people.groupby("model", sort=False):
        fold = folds[folds.model == name]
        spread = seeds_frame[seeds_frame.model == name].groupby("fold").equal_person_balanced_accuracy.std(ddof=1)
        summary.append({"model": name, "people": len(own), "eligible_people": int(own.two_class_defined.sum()),
                        "equal_person_balanced_accuracy": float(own.balanced_accuracy.mean()),
                        "person_ba_sd": float(own.balanced_accuracy.std(ddof=1)),
                        "equal_person_macro_f1": float(own.macro_f1.mean()),
                        "outer_fold_ba_sd": float(fold.equal_person_balanced_accuracy.std(ddof=1)),
                        "fold_ba_min": float(fold.equal_person_balanced_accuracy.min()),
                        "fold_ba_max": float(fold.equal_person_balanced_accuracy.max()),
                        "seed_ba_sd_mean_over_folds": float(spread.mean()),
                        "tuning_total_seconds": float(fold.tuning_seconds.sum()),
                        "refit_median_seconds_across_folds": float(fold.fit_seconds.median()),
                        "predict_median_seconds_across_folds": float(fold.predict_median_seconds.median()),
                        "parameters_total_median": float(fold.parameters_total.median()),
                        "pooled": metrics(pred[pred.model == name].target, pred[pred.model == name].predicted)})
    # Independent recomputation and inner-only selection.
    for (model, pid), rows in pred.groupby(["model", "pid"]):
        own = people[(people.model == model) & (people.pid == pid)].iloc[0]
        assert tuple(confusion_matrix(rows.target, rows.predicted, labels=[0, 1]).ravel()) == (own.tn, own.fp, own.fn, own.tp)
        if rows.target.nunique() == 2:
            assert np.isclose(balanced_accuracy_score(rows.target, rows.predicted), own.balanced_accuracy)
    search = pd.DataFrame(searches)
    for row in folds.itertuples():
        own = search[(search.model == row.model) & (search.outer_fold == row.fold)]
        best = json.loads(own.loc[own.inner_equal_person_balanced_accuracy.idxmax(), "parameters"])
        assert all(json.loads(row.selected_parameters)[k] == v for k, v in best.items())
    output = A3 / "results/deep_v3"
    folds.to_csv(output / "fold_metrics.csv", index=False)
    people.to_csv(output / "participant_metrics.csv", index=False)
    pred.to_csv(output / "predictions.csv", index=False)
    search.to_csv(output / "search.csv", index=False)
    seeds_frame.to_csv(output / "seeds.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(output / "timing.csv", index=False)
    pd.DataFrame([{k: v for k, v in row.items() if k != "pooled"} for row in summary]).to_csv(output / "summary.csv", index=False)
    write_json(output / "summary.json", summary)
    write_json(output / "verification.json", {"status": "PASS", "prediction_rows": len(pred), "participant_metric_rows": len(people),
                                              "independent_sklearn_metrics_and_confusions": "PASS",
                                              "selected_parameters_match_inner_search_only": "PASS",
                                              "locked_test_predictions": 0})
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    write_json(output / "environment.json", {
        "command": ".venv/bin/python -B assignment3/src/cnn.py", "python": sys.version,
        "platform": platform.platform(), "cpu": platform.processor(), "logical_cpus": os.cpu_count(),
        "torch_training_threads": TRAIN_THREADS, "torch_prediction_timing_threads": 1, "packages": packages,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT).decode(),
        "source_sha256": sha256(__file__), "plan_sha256": sha256(PLAN), "sequences": evidence["code_sha256"],
        "timing_method": "refit wall time with 4 threads; prediction in 5 warm single-thread batches (>=20 ms or 1000 calls)",
        "full_development_final_fit": "not performed; selected outer refits only", "test_predictions": "not performed"})
    write_json(output / "checks.json", {"status": "PASS", "development_windows": len(meta), "people": int(meta.pid.nunique()),
                                        "predictions": len(pred), "search_candidates": len(search),
                                        "checks": ["sequence and source digests", "disjoint nested person folds",
                                                   "no locked-test person in target or source data",
                                                   "pretraining only on the partition's training people",
                                                   "pupil statistics from training windows only",
                                                   "B sequences centred per window", "unique OOF predictions",
                                                   "independent sklearn recomputation", "inner-only selection"]})
    print(pd.read_csv(output / "summary.csv")[["model", "equal_person_balanced_accuracy", "person_ba_sd", "fold_ba_min",
                                              "fold_ba_max", "seed_ba_sd_mean_over_folds"]].to_string(index=False))


if __name__ == "__main__":
    main()
