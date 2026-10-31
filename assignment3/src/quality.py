"""Audit time support; optionally apply a declared quality policy to V1 windows."""
import argparse
import json

import numpy as np
import pandas as pd

from data import A3, ROOT, CACHE_COLUMNS, CODE_PATTERN, counts, read_config, sha256, write_json
from checks import run_checks
import outliers


def time_parts(times, quality, start, end, period_us=4000):
    """Partition the same conservative sample support as V1, by measurement label."""
    if len(times) == 0:
        return 0.0, 0.0, (end-start)/1e6
    assert (np.diff(times) > 0).all() and times.min() >= start and times.max() < end
    assert set(quality) <= {"ok", "interpolated"}
    support = np.minimum(times+period_us, np.r_[times[1:], end])-times
    measured = float(support[quality == "ok"].sum()) / 1e6
    interpolated = float(support[quality == "interpolated"].sum()) / 1e6
    uncovered = (end-start)/1e6 - measured - interpolated
    assert uncovered >= -1e-12
    return measured, interpolated, max(0.0, uncovered)


def distribution(frame):
    result = {"windows": len(frame), "people": int(frame.pid.nunique())}
    for col in ["measured_fraction", "interpolated_time_fraction", "uncovered_fraction"]:
        result[col] = {str(q): float(frame[col].quantile(q))
                       for q in [0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1]}
    result["total_seconds"] = {col: float(frame[col].sum())
                               for col in ["measured_seconds", "interpolated_seconds", "uncovered_seconds"]}
    result["interpolation_bins"] = {
        "zero": int((frame.interpolated_time_fraction == 0).sum()),
        "over_0_to_5_percent": int(frame.interpolated_time_fraction.between(0, .05, inclusive="right").sum()),
        "over_5_to_10_percent": int(frame.interpolated_time_fraction.between(.05, .10, inclusive="right").sum()),
        "over_10_to_20_percent": int(frame.interpolated_time_fraction.between(.10, .20, inclusive="right").sum()),
        "over_20_percent": int((frame.interpolated_time_fraction > .20).sum()),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Apply quality_policy.json after reviewing the audit")
    args = parser.parse_args()
    run_checks()
    config = read_config()
    output = A3 / "results/quality"
    output.mkdir(exist_ok=True)
    df = pd.read_parquet(ROOT / config["a1_cache"], columns=CACHE_COLUMNS)
    for col in ["pid", "stimulus", "quality"]: df[col] = df[col].astype(str)
    df = df[df.stimulus.str.match(CODE_PATTERN)].copy()
    df = df.loc[~(outliers.velocity(df) > outliers.MAX_DEG_S)]
    groups = dict(tuple(df.groupby(["pid", "segment"], sort=False)))
    audit = pd.read_csv(A3 / "results/data/window_audit.csv", dtype={"pid": str})
    rows = []
    for window in audit.itertuples():
        samples = groups[(window.pid, window.segment)]
        times = samples.Time.to_numpy()
        left, right = np.searchsorted(times, [window.start_us, window.end_us])
        measured, interpolated, uncovered = time_parts(
            times[left:right], samples.quality.to_numpy()[left:right], window.start_us, window.end_us)
        duration = (window.end_us-window.start_us)/1e6
        assert np.isclose(measured+interpolated, window.supported_seconds, atol=1e-9)
        rows.append({"pid": window.pid, "segment": window.segment, "window_index": window.window_index,
                     "target": window.target, "split": window.split, "accepted_v1": window.accepted,
                     "measured_seconds": measured, "interpolated_seconds": interpolated,
                     "uncovered_seconds": uncovered, "measured_fraction": measured/duration,
                     "interpolated_time_fraction": interpolated/duration, "uncovered_fraction": uncovered/duration})
    detail = pd.DataFrame(rows)
    assert np.allclose(detail.measured_fraction+detail.interpolated_time_fraction+detail.uncovered_fraction, 1)
    # Independently check mixed support and an absent 200 ms interval.
    measured, interpolated, uncovered = time_parts(
        np.array([0, 4000, 208000]), np.array(["ok", "interpolated", "ok"]), 0, 212000)
    assert np.allclose([measured, interpolated, uncovered], [.008, .004, .200])
    development = detail[detail.split == "development"]
    summary = {"scope": "Development quality only; no model scores inspected",
               "v1_development_accepted": distribution(development[development.accepted_v1]),
               "v1_development_rejected": distribution(development[~development.accepted_v1]),
               "checks": "three time parts sum to window duration; support matches V1; synthetic mixed/missing support passes",
               "quality_source_sha256": sha256(__file__),
               "v1_windows_sha256": sha256(A3 / "cache/windows.parquet")}
    detail.to_csv(output / "window_time_parts.csv", index=False)
    write_json(output / "audit_summary.json", summary)
    if args.apply:
        policy = json.loads((A3 / "quality_policy.json").read_text())
        detail["accepted_v2"] = detail.accepted_v1 & (detail.measured_fraction >= policy["min_measured_fraction"])
        detail.to_csv(output / "window_time_parts.csv", index=False)
        original = pd.read_parquet(A3 / "cache/windows.parquet")
        updated = original.merge(detail[["pid", "segment", "window_index", "measured_fraction",
                                          "interpolated_time_fraction", "uncovered_fraction", "accepted_v2"]],
                                 on=["pid", "segment", "window_index"], validate="one_to_one")
        updated = updated[updated.accepted_v2].copy()
        updated.to_parquet(A3 / "cache/windows_v2.parquet", index=False)
        all_parts = {}
        for name in ["development", "locked_test"]:
            before, after = original[original.split == name], updated[updated.split == name]
            all_parts[name] = {"before": counts(before), "after": counts(after),
                               "lost_people": sorted(set(before.pid)-set(after.pid), key=int),
                               "single_class_people": sorted(after.groupby("pid").target.nunique().loc[lambda s:s<2].index, key=int)}
        by_person = detail.groupby(["split", "pid", "target"], as_index=False).agg(
            candidates=("window_index", "size"), accepted_v1=("accepted_v1", "sum"),
            accepted_v2=("accepted_v2", "sum"), measured_seconds=("measured_seconds", "sum"),
            interpolated_seconds=("interpolated_seconds", "sum"), uncovered_seconds=("uncovered_seconds", "sum"))
        by_person.to_csv(output / "participant_class_quality.csv", index=False)
        selection = {"policy": policy, "subsets": all_parts, "v2_windows_sha256": sha256(A3 / "cache/windows_v2.parquet"),
                     "quality_source_sha256": sha256(__file__), "test_predictions": "not performed"}
        write_json(output / "selection.json", selection)
        print(json.dumps(selection, indent=2))
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
