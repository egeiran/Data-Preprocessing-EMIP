"""V3 window features: I-VT fixations and saccades from coordinate differences only."""
import hashlib
import json

import numpy as np
import pandas as pd

from data import A3, ROOT, CACHE_COLUMNS, CODE_PATTERN, FEATURE_GROUPS, read_config, sha256, write_json
import outliers

PLAN = A3 / "feature_plan_v3.json"
KEY = ["pid", "segment", "window_index"]
EVENT_FEATURES = [
    "fixation_rate_per_s", "fixation_duration_mean_ms", "fixation_time_fraction",
    "saccade_rate_per_s", "saccade_amplitude_mean_deg", "horizontal_saccade_fraction",
    "leftward_saccade_fraction", "upward_saccade_fraction",
]
FEATURE_GROUPS_V3 = {"A3_with_position": FEATURE_GROUPS["A_with_position"] + EVENT_FEATURES,
                     "B3_without_absolute_position": FEATURE_GROUPS["B_without_absolute_position"] + EVENT_FEATURES}
PERIOD_MS = 4.0


def fingerprint(frame, columns):
    """Content hash that is stable across machines. Parquet bytes are not: the same
    rows written on macOS and on Linux/WSL hash differently (see WORKLOG)."""
    ordered = frame.sort_values(KEY)[KEY + list(columns)]
    text = ordered.to_csv(index=False, float_format="%.17g", lineterminator="\n")
    return hashlib.sha256(text.encode()).hexdigest()


def events(times, x, y, measured, plan):
    """I-VT on one window. A sample is classified only when it and both neighbours are
    measured and both intervals are short; anything else ends the current event."""
    n = len(times)
    label = np.zeros(n, dtype=int)  # 0 unclassified, 1 fixation, 2 saccade
    max_us = plan["max_sample_interval_ms"] * 1000
    if n >= 3:
        dt = np.diff(times)
        ok = (dt[:-1] <= max_us) & (dt[1:] <= max_us) & measured[:-2] & measured[1:-1] & measured[2:]
        span = (times[2:] - times[:-2]) / 1e6
        speed = np.hypot(x[2:] - x[:-2], y[2:] - y[:-2]) / outliers.PX_PER_DEG / span
        label[1:-1] = np.where(ok, np.where(speed < plan["velocity_threshold_deg_s"], 1, 2), 0)
    breaks = np.r_[True, (label[1:] != label[:-1]) | (np.diff(times) > max_us)]
    starts = np.flatnonzero(breaks)
    ends = np.r_[starts[1:], n] - 1
    fixations, saccades = [], []
    for s, e in zip(starts, ends):
        if label[s] == 1:
            duration = (times[e] - times[s]) / 1000 + PERIOD_MS
            if duration >= plan["min_fixation_ms"]:
                fixations.append(duration)
        elif label[s] == 2 and e - s + 1 >= plan["min_saccade_samples"]:
            # Centred speed: the movement runs from the neighbour before to the one after.
            saccades.append((x[e + 1] - x[s - 1], y[e + 1] - y[s - 1]))
    return fixations, saccades


def event_features(fixations, saccades, seconds):
    fix = np.asarray(fixations, dtype=float)
    sac = np.asarray(saccades, dtype=float).reshape(-1, 2)
    horizontal = np.abs(sac[:, 0]) >= np.abs(sac[:, 1])
    every = np.ones(len(sac), dtype=bool)

    def share(mask, among):
        return float(mask[among].mean()) if among.any() else 0.0

    return {"fixation_rate_per_s": len(fix) / seconds,
            "fixation_duration_mean_ms": float(fix.mean()) if len(fix) else 0.0,
            "fixation_time_fraction": float(fix.sum()) / (seconds * 1000),
            "saccade_rate_per_s": len(sac) / seconds,
            "saccade_amplitude_mean_deg": float(np.hypot(*sac.T).mean() / outliers.PX_PER_DEG) if len(sac) else 0.0,
            "horizontal_saccade_fraction": share(horizontal, every),
            "leftward_saccade_fraction": share(sac[:, 0] < 0, horizontal),
            "upward_saccade_fraction": share(~horizontal & (sac[:, 1] < 0), every)}


def window_features(part, seconds, plan, shift=(0.0, 0.0)):
    fixations, saccades = events(part.Time.to_numpy(dtype=np.int64),
                                 part["L POR X [px]"].to_numpy() + shift[0],
                                 part["L POR Y [px]"].to_numpy() + shift[1],
                                 part.quality.to_numpy() == "ok", plan)
    return {**event_features(fixations, saccades, seconds),
            "fixation_count": len(fixations), "saccade_count": len(saccades)}


def run_fixation_checks(plan):
    """Synthetic signals with known events, in the style of checks.py."""
    t = np.arange(0, 2_000_000, 4000, dtype=np.int64)
    n, ppd = len(t), outliers.PX_PER_DEG
    flat, measured = np.full(n, 500.0), np.ones(n, dtype=bool)
    # A still gaze is one fixation; the first and last sample have no centred speed.
    fix, sac = events(t, flat, flat, measured, plan)
    assert fix == [1992.0] and sac == []
    # A 5 degree, 20 ms step splits two fixations and is one saccade of 5 degrees.
    step = np.interp(t, [0, 300_000, 320_000, 2_000_000], [500, 500, 500 + 5 * ppd, 500 + 5 * ppd])
    fix, sac = events(t, step, flat, measured, plan)
    assert len(fix) == 2 and len(sac) == 1 and np.isclose(np.hypot(*sac[0]) / ppd, 5)
    right = event_features(fix, sac, 2.0)
    assert right["horizontal_saccade_fraction"] == 1 and right["leftward_saccade_fraction"] == 0
    left = event_features(*events(t, 1000 - step, flat, measured, plan), 2.0)
    assert left["leftward_saccade_fraction"] == 1 and left["upward_saccade_fraction"] == 0
    up = event_features(*events(t, flat, 1000 - step, measured, plan), 2.0)
    assert up["horizontal_saccade_fraction"] == 0 and up["upward_saccade_fraction"] == 1
    # A 100 ms hole ends a fixation; it is not compressed into continuous time.
    keep = (t < 1_000_000) | (t >= 1_100_000)
    fix, _ = events(t[keep], flat[keep], flat[keep], measured[keep], plan)
    assert len(fix) == 2
    # An interpolated sample ends a fixation; the 4 ms piece before it is too short.
    gap = measured.copy()
    gap[3] = False
    fix, _ = events(t, flat, flat, gap, plan)
    assert len(fix) == 1
    empty = event_features([], [], 2.0)
    assert all(np.isfinite(v) and v == 0 for v in empty.values())
    return ["still gaze", "5 degree step", "direction shares", "hole ends fixation",
            "interpolated sample ends fixation", "minimum fixation duration", "empty-window fill"]


def load_v2(selection):
    """Portable V2 check: keys against committed CSV evidence, not parquet bytes."""
    v2 = pd.read_parquet(A3 / "cache/windows_v2.parquet")
    parts = pd.read_csv(A3 / "results/quality/window_time_parts.csv", dtype={"pid": str})
    expected = parts.loc[parts.accepted_v2, KEY].sort_values(KEY).reset_index(drop=True)
    assert v2[KEY].sort_values(KEY).reset_index(drop=True).equals(expected)
    assert not v2.duplicated(KEY).any()
    for split, subset in selection["subsets"].items():
        assert (v2.split == split).sum() == subset["after"]["windows"]
    assert v2.measured_fraction.ge(selection["policy"]["min_measured_fraction"]).all()
    assert set(v2.loc[v2.split == "locked_test", "pid"]) == set(selection["policy"]["test_participants"])
    return v2


def main():
    plan = json.loads(PLAN.read_text())
    assert plan["feature_groups"] == FEATURE_GROUPS_V3
    position = set(FEATURE_GROUPS_V3["A3_with_position"]) - set(FEATURE_GROUPS_V3["B3_without_absolute_position"])
    assert position == {"gaze_x_mean_px", "gaze_y_mean_px"}
    synthetic = run_fixation_checks(plan)
    selection = json.loads((A3 / "results/quality/selection.json").read_text())
    v2 = load_v2(selection)
    config = read_config()
    df = pd.read_parquet(ROOT / config["a1_cache"], columns=CACHE_COLUMNS)
    for col in ["pid", "stimulus", "quality"]:
        df[col] = df[col].astype(str)
    df = df[df.stimulus.str.match(CODE_PATTERN)].copy()
    df = df.loc[~(outliers.velocity(df) > outliers.MAX_DEG_S)]
    groups = dict(tuple(df.groupby(["pid", "segment"], sort=False)))
    rows, shifted_equal = [], 0
    for window in v2.itertuples():
        samples = groups[(window.pid, window.segment)]
        left, right = np.searchsorted(samples.Time.to_numpy(), [window.start_us, window.end_us])
        part = samples.iloc[left:right]
        assert len(part) == window.n_samples
        seconds = (window.end_us - window.start_us) / 1e6
        row = window_features(part, seconds, plan)
        # Moving the whole window on screen must not change any event feature.
        moved = window_features(part, seconds, plan, shift=(128.0, -64.0))
        assert row["fixation_count"] == moved["fixation_count"] and row["saccade_count"] == moved["saccade_count"]
        assert np.allclose([row[f] for f in EVENT_FEATURES], [moved[f] for f in EVENT_FEATURES], rtol=1e-9, atol=1e-9)
        shifted_equal += 1
        rows.append({**{k: getattr(window, k) for k in KEY}, **row})
    v3 = v2.merge(pd.DataFrame(rows), on=KEY, validate="one_to_one")
    assert len(v3) == len(v2) and np.isfinite(v3[FEATURE_GROUPS_V3["A3_with_position"]]).all().all()
    output = A3 / "results/features_v3"
    output.mkdir(parents=True, exist_ok=True)
    v3.to_parquet(A3 / "cache/windows_v3.parquet", index=False)
    development = v3[v3.split == "development"]
    quantiles = [0, .05, .25, .5, .75, .95, 1]
    summary = {"scope": "Development distributions only; no labels or model scores inspected",
               "windows": {s: int((v3.split == s).sum()) for s in ["development", "locked_test"]},
               "people": {s: int(v3.loc[v3.split == s, "pid"].nunique()) for s in ["development", "locked_test"]},
               "windows_without_fixation": {s: int(((v3.split == s) & (v3.fixation_count == 0)).sum()) for s in ["development", "locked_test"]},
               "windows_without_saccade": {s: int(((v3.split == s) & (v3.saccade_count == 0)).sum()) for s in ["development", "locked_test"]},
               "development_quantiles": {f: {str(q): float(development[f].quantile(q)) for q in quantiles}
                                         for f in EVENT_FEATURES + ["fixation_count", "saccade_count"]}}
    provenance = {"plan_sha256": sha256(PLAN), "source_sha256": sha256(__file__),
                  "v2_content_sha256": fingerprint(v2, FEATURE_GROUPS["A_with_position"] + ["target", "split"]),
                  "v3_content_sha256": fingerprint(v3, FEATURE_GROUPS_V3["A3_with_position"] + ["target", "split"]),
                  "fingerprint": "sha256 of KEY-sorted CSV text with %.17g floats; portable across machines"}
    write_json(output / "summary.json", summary)
    write_json(output / "provenance.json", provenance)
    write_json(output / "checks.json", {"status": "PASS", "windows": len(v3), "translation_checked_windows": shifted_equal,
                                        "checks": synthetic + ["V2 keys match committed quality evidence",
                                                               "V2 counts and locked people match selection.json",
                                                               "sample counts match V2 windows",
                                                               "every real window unchanged by a screen shift",
                                                               "finite V3 features", "B3 has no absolute position"]})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
