"""Build auditable clock-based windows; never write to an A1 path."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import cleaning
import outliers
import transform

A3 = ROOT / "assignment3"
CODE_PATTERN = r"^(vehicle|rectangle)_(java|java2|scala)\.jpg$"
POSITION = ["gaze_x_mean_px", "gaze_y_mean_px"]
WITHOUT_POSITION = [
    "gaze_x_std_px", "gaze_y_std_px",
    "pupil_left_mean_mm", "pupil_left_std_mm",
    "pupil_right_mean_mm", "pupil_right_std_mm",
    "velocity_median_deg_s", "velocity_p90_deg_s",
    "a1_blink_rate_mean", "a1_gap_fraction_mean", "interpolated_fraction",
]
FEATURE_GROUPS = {"A_with_position": POSITION + WITHOUT_POSITION,
                  "B_without_absolute_position": WITHOUT_POSITION}
SIGNALS = ["L POR X [px]", "L POR Y [px]", "L Mapped Diameter [mm]",
           "R Mapped Diameter [mm]", "blink_rate", "gap_fraction"]
CACHE_COLUMNS = ["pid", "segment", "Time", "stimulus", "quality"] + SIGNALS


def read_config():
    return json.loads((A3 / "config.json").read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def coverage(times, start, end, period_us):
    """Conservative time support: each in-window sample covers at most one period.

    Clip at the next sample and window end, so jitter cannot double-count time.
    Initial empty time, internal holes and the final empty tail all count as gaps.
    Interpolated samples count as support, and are flagged separately.
    """
    if len(times) == 0:
        return 0.0, (end - start) / 1000, 0.0
    if not ((times >= start).all() and (times < end).all()
            and (np.diff(times) > 0).all()):
        raise ValueError("Unsorted, duplicate or out-of-window timestamp")
    stops = np.minimum(times + period_us, np.r_[times[1:], end])
    supported_us = float((stops - times).sum())
    gaps = np.r_[times[0] - start, times[1:] - stops[:-1], end - stops[-1]]
    return supported_us / (end - start), float(gaps.max()) / 1000, supported_us / 1e6


def features(frame, config):
    """Row-weighted mean/population SD; speed uses in-window pairs only."""
    result = {}
    for source, prefix, unit in [
        ("L POR X [px]", "gaze_x", "px"), ("L POR Y [px]", "gaze_y", "px"),
        ("L Mapped Diameter [mm]", "pupil_left", "mm"),
        ("R Mapped Diameter [mm]", "pupil_right", "mm"),
    ]:
        values = frame[source].to_numpy()
        result[f"{prefix}_mean_{unit}"] = float(values.mean())
        result[f"{prefix}_std_{unit}"] = float(values.std(ddof=0))
    dt = np.diff(frame.Time.to_numpy()) / 1e6
    speed = np.hypot(np.diff(frame["L POR X [px]"]), np.diff(frame["L POR Y [px]"]))
    speed = speed / outliers.PX_PER_DEG / dt
    valid = ((dt > 0) & (dt <= config["max_velocity_interval_ms"] / 1000)
             & np.isfinite(speed) & (speed <= outliers.MAX_DEG_S))
    if not valid.any():
        raise ValueError("Accepted window has no valid within-window velocity")
    result["velocity_median_deg_s"] = float(np.median(speed[valid]))
    result["velocity_p90_deg_s"] = float(np.quantile(speed[valid], .90))
    result["a1_blink_rate_mean"] = float(frame.blink_rate.mean())
    result["a1_gap_fraction_mean"] = float(frame.gap_fraction.mean())
    result["interpolated_fraction"] = float((frame.quality == "interpolated").mean())
    return result


def counts(frame):
    return {"windows": len(frame), "participants": int(frame.pid.nunique()),
            "displays": int(frame.groupby(["pid", "segment"]).ngroups),
            "classes": {str(k): int(v) for k, v in frame.target.value_counts().items()},
            "nominal_window_seconds": float((frame.end_us - frame.start_us).sum() / 1e6),
            "supported_seconds": float(frame.supported_seconds.sum())}


def main():
    config = read_config()
    if config["pupil_clipping"] != "none":
        raise ValueError("Only the documented no-clipping first experiment is implemented")
    cache = ROOT / config["a1_cache"]
    if not cache.is_file():
        raise SystemExit("Missing A1 cache; rebuild it separately. A3 will not write it.")
    output = A3 / "results/data"
    output.mkdir(parents=True, exist_ok=True)
    protected = [ROOT / p for p in subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0") if p
        and p not in ("README.md", ".gitignore", "IT3212-2026-Assignment 3 (1).pdf")
        and not p.startswith("assignment3/")]
    protected.append(cache)
    snapshot = {str(p.relative_to(ROOT)): sha256(p) for p in protected}
    provenance = {"a1_reference_commit": config["a1_reference_commit"],
                  "protected_a1_sha256": snapshot,
                  "a3_pdf_sha256": sha256(A3 / "assignment.pdf"),
                  "config": config,
                  "source_sha256": {str(p.relative_to(ROOT)): sha256(p)
                                    for p in sorted((A3 / "src").glob("*.py"))}}
    df = pd.read_parquet(cache, columns=CACHE_COLUMNS)
    for name in ["pid", "stimulus", "quality"]:
        df[name] = df[name].astype(str)
    # Reconstruct A1's original modelling participant split, including choice screens.
    a1_meta = transform.label(df[["pid", "stimulus"]])
    _, a1_y, a1_groups, _ = transform.encode(a1_meta.assign(
        quality="ok", **{"Pupil Confidence": 1}))
    a1_split = transform.split(pd.DataFrame(index=a1_y.index), a1_y, a1_groups)
    reconstructed_test = set(a1_split[-1])
    if reconstructed_test != set(config["a1_test_participants"]):
        raise ValueError("Configured locked test differs from original A1 participant split")
    df = df[df.stimulus.str.match(CODE_PATTERN)].copy()
    before_rows = len(df)
    v = outliers.velocity(df)
    removed = v > outliers.MAX_DEG_S
    source_counts = {"rows_before_velocity_filter": before_rows,
                     "rows_removed_by_velocity_filter": int(removed.sum()),
                     "participants": int(df.pid.nunique())}
    timestamp_delta = df.groupby(["pid", "segment"]).Time.diff() / 1000
    source_counts["retained_timestamp_gap_ms_quantiles"] = {
        str(q): float(timestamp_delta.quantile(q)) for q in [.5, .99, .999, 1]}
    source_counts["retained_intervals_over_150_ms"] = int((timestamp_delta > 150).sum())
    # Retain unfiltered cache for exact comparison with A1 preprocessing below.
    cached_groups = {key: group for key, group in df.groupby(["pid", "segment"], sort=False)}
    retained = df.loc[~removed].copy()
    retained_groups = {key: group for key, group in retained.groupby(["pid", "segment"], sort=False)}
    source_counts["rows_after_velocity_filter"] = len(retained)
    source_counts["usable_sample_equivalent_seconds"] = len(retained) / 250
    source_counts["class_rows_after_velocity_filter"] = retained.stimulus.str.split("_").str[0].value_counts().to_dict()
    source_counts["pupil_range_mm_without_clipping"] = {
        c: {"min": float(retained[c].min()), "max": float(retained[c].max())}
        for c in SIGNALS[2:4]}
    windows, audits, displays, recordings = [], [], [], []
    checked_keys = set()
    window_us = int(config["window_seconds"] * 1e6)
    raw_paths = [ROOT / p for p in cleaning.all_paths() if (ROOT / p).is_file()]
    for path in raw_paths:
        pid = path.name.split("_")[0]
        raw = cleaning.load(path)
        rate = raw.attrs["sample_rate"]
        if rate != 250:
            raise ValueError("This first-run configuration requires the checked 250 Hz recordings")
        period_us = int(1e6 / rate)
        processed, _ = cleaning.preprocess(raw)
        seconds = cleaning.code_seconds(processed)
        text = raw["L Raw X [px]"].astype(str).str.replace("# Message: ", "", regex=False)
        messages = raw.loc[(raw.Type == "MSG") & text.str.endswith(".jpg"), ["Time"]].copy()
        messages["stimulus"] = text.loc[messages.index]
        messages["segment"] = np.arange(1, len(messages) + 1)
        final_end = int(raw.loc[raw.Type == "SMP", "Time"].max()) + period_us
        messages["end_us"] = np.r_[messages.Time.to_numpy()[1:], final_end]
        recordings.append({"pid": pid, "file": str(path.relative_to(ROOT)),
                           "sha256": sha256(path), "sample_rate_hz": rate,
                           "raw_samples": int((raw.Type == "SMP").sum()),
                           "raw_messages": int((raw.Type == "MSG").sum())})
        for msg in messages.itertuples():
            if not re.fullmatch(CODE_PATTERN, msg.stimulus):
                continue
            key = (pid, msg.segment)
            target = int(msg.stimulus.startswith("vehicle_"))
            group = cached_groups.get(key)
            expected = processed[(processed.segment == msg.segment) & (processed.quality != "gap")]
            a1_kept = seconds.get(msg.stimulus.split("_")[0], 0) >= cleaning.MIN_CODE_S
            if a1_kept:
                if group is None or not np.array_equal(group.Time.to_numpy(), expected.Time.to_numpy()):
                    raise ValueError(f"A1 cache times disagree with current preprocessing: {key}")
                if not np.allclose(group[SIGNALS], expected[SIGNALS], rtol=0, atol=1e-10):
                    raise ValueError(f"A1 cache signals disagree with current preprocessing: {key}")
                if not np.array_equal(group.quality.to_numpy(), expected.quality.to_numpy()):
                    raise ValueError(f"A1 quality labels disagree: {key}")
                checked_keys.add(key)
            elif group is not None:
                raise ValueError(f"A1 cache retained an excluded display: {key}")
            valid = retained_groups.get(key, pd.DataFrame(columns=CACHE_COLUMNS))
            start, end = int(msg.Time), int(msg.end_us)
            times = valid.Time.to_numpy(dtype=np.int64)
            if len(times) and not ((np.diff(times) > 0).all()
                                   and times.min() >= start and times.max() < end):
                raise ValueError(f"Invalid screen boundaries/timestamp order: {key}")
            display = {"pid": pid, "segment": msg.segment, "stimulus": msg.stimulus,
                       "target": target, "split": "locked_test" if pid in reconstructed_test else "development",
                       "start_us": start, "end_us": end, "screen_seconds": (end-start)/1e6,
                       "a1_usable_seconds_before_velocity_filter": float(seconds.get(msg.stimulus.split("_")[0], 0)),
                       "a1_retained": bool(a1_kept), "rows_after_velocity_filter": len(valid),
                       "candidate_windows": 0, "accepted_windows": 0,
                       "trailing_incomplete_seconds": 0.0}
            if a1_kept:
                n_windows = (end - start) // window_us
                display["candidate_windows"] = n_windows
                display["trailing_incomplete_seconds"] = (end-start-n_windows*window_us)/1e6
                for index in range(n_windows):
                    left, right = start + index*window_us, start + (index+1)*window_us
                    lo, hi = np.searchsorted(times, [left, right])
                    part = valid.iloc[lo:hi]
                    fraction, gap_ms, supported_s = coverage(times[lo:hi], left, right, period_us)
                    reasons = []
                    if fraction < config["min_coverage"]: reasons.append("low_coverage")
                    if gap_ms > config["max_uncovered_ms"]: reasons.append("long_uncovered_interval")
                    row = {"pid": pid, "segment": msg.segment, "window_index": index,
                           "stimulus": msg.stimulus, "target": target, "split": display["split"],
                           "start_us": left, "end_us": right, "n_samples": len(part),
                           "coverage": fraction, "max_uncovered_ms": gap_ms,
                           "supported_seconds": supported_s, "accepted": not reasons,
                           "reason": "+".join(reasons)}
                    audits.append(row)
                    if not reasons:
                        windows.append({**row, **features(part, config)})
                        display["accepted_windows"] += 1
            displays.append(display)
        print(f"Checked raw/cache participant {pid}; {len(windows)} accepted windows so far", flush=True)
    if checked_keys != set(cached_groups):
        raise ValueError("Some cached code displays have no verified raw source")
    result = pd.DataFrame(windows)
    if not np.isfinite(result[FEATURE_GROUPS["A_with_position"]]).all().all():
        raise ValueError("Non-finite features")
    (A3 / "cache").mkdir(exist_ok=True)
    result.to_parquet(A3 / "cache/windows.parquet", index=False)
    pd.DataFrame(audits).to_csv(output / "window_audit.csv", index=False)
    pd.DataFrame(displays).to_csv(output / "displays.csv", index=False)
    pd.DataFrame(recordings).to_csv(output / "recordings.csv", index=False)
    summary = {"source": source_counts, "raw_recordings": len(recordings),
               "raw_samples": sum(r["raw_samples"] for r in recordings),
               "raw_messages": sum(r["raw_messages"] for r in recordings),
               "raw_code_displays": len(displays), "cache_verified_displays": len(checked_keys),
               "candidate_windows": len(audits), "rejected_windows": len(audits)-len(result),
               "rejection_reasons": pd.DataFrame(audits).loc[lambda d: ~d.accepted, "reason"].value_counts().to_dict(),
               "feature_groups": FEATURE_GROUPS, "all": counts(result),
               "development": counts(result[result.split == "development"]),
               "locked_test": counts(result[result.split == "locked_test"]),
               "participants_without_accepted_windows": sorted(set(retained.pid)-set(result.pid))}
    provenance["windows_sha256"] = sha256(A3 / "cache/windows.parquet")
    write_json(output / "provenance.json", provenance)
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
