"""Targeted data-integrity checks, including synthetic missing-time examples."""
import json

import numpy as np
import pandas as pd

from data import A3, ROOT, FEATURE_GROUPS, coverage, read_config, sha256, write_json


def run_checks():
    config = read_config()
    provenance = json.loads((A3 / "results/data/provenance.json").read_text())
    if provenance["config"] != config:
        raise ValueError("Config differs from the built data; rebuild before running")
    for name, expected in provenance["protected_a1_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Protected A1 file changed: {name}")
    if sha256(A3 / "cache/windows.parquet") != provenance["windows_sha256"]:
        raise ValueError("Window table differs from its provenance")
    if sha256(A3 / "assignment.pdf") != provenance["a3_pdf_sha256"]:
        raise ValueError("A3 assignment PDF changed")
    for name, expected in provenance["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Builder/check/experiment code changed since data build: {name}")
    windows = pd.read_parquet(A3 / "cache/windows.parquet")
    audit = pd.read_csv(A3 / "results/data/window_audit.csv", dtype={"pid": str})
    displays = pd.read_csv(A3 / "results/data/displays.csv", dtype={"pid": str})
    test = set(config["a1_test_participants"])
    development = set(windows.loc[windows.split == "development", "pid"])
    locked = set(windows.loc[windows.split == "locked_test", "pid"])
    assert not (development & test) and locked == test
    assert (windows.split == np.where(windows.pid.isin(test), "locked_test", "development")).all()
    key = ["pid", "segment", "window_index"]
    assert not windows.duplicated(key).any()
    assert not audit.duplicated(key).any()
    assert windows[key].equals(audit.loc[audit.accepted, key].reset_index(drop=True))
    joined = audit.merge(displays, on=["pid", "segment"], suffixes=("", "_display"), validate="many_to_one")
    assert len(joined) == len(audit)
    duration = int(config["window_seconds"] * 1e6)
    assert (joined.end_us - joined.start_us == duration).all()
    assert (joined.start_us == joined.start_us_display + joined.window_index * duration).all()
    assert (joined.end_us <= joined.end_us_display).all()
    assert (joined.stimulus == joined.stimulus_display).all()
    assert (joined.target == joined.target_display).all()
    assert (joined.target == joined.stimulus.str.startswith("vehicle_").astype(int)).all()
    assert joined.stimulus.str.match(r"^(vehicle|rectangle)_(java|java2|scala)\.jpg$").all()
    ordered = audit.sort_values(key)
    previous_end = ordered.groupby(["pid", "segment"]).end_us.shift()
    assert (ordered.start_us >= previous_end).fillna(True).where(previous_end.notna(), True).all()
    expected_accept = ((audit.coverage >= config["min_coverage"])
                       & (audit.max_uncovered_ms <= config["max_uncovered_ms"]))
    assert (audit.accepted == expected_accept).all()
    assert windows.n_samples.gt(0).all()
    assert windows.coverage.between(config["min_coverage"], 1 + 1e-12).all()
    assert np.allclose(windows.supported_seconds, windows.coverage * config["window_seconds"])
    assert windows.max_uncovered_ms.le(config["max_uncovered_ms"]).all()
    assert windows.interpolated_fraction.between(0, 1).all()
    forbidden = {"pid", "segment", "window_index", "stimulus", "target", "split", "start_us", "end_us",
                 "file", "language", "Time", "coverage", "n_samples", "accepted", "reason"}
    for names in FEATURE_GROUPS.values():
        assert len(names) == len(set(names))
        assert not (set(names) & forbidden)
        assert np.isfinite(windows[names].to_numpy(dtype=float)).all()
        assert not any(name.startswith(("L Raw", "R Raw", "L EPOS", "R EPOS", "L GVEC", "R GVEC")) for name in names)
    assert set(FEATURE_GROUPS["A_with_position"]) - set(FEATURE_GROUPS["B_without_absolute_position"]) == {"gaze_x_mean_px", "gaze_y_mean_px"}
    # A 200 ms deleted run must not look like 450 consecutive 250 Hz samples.
    regular = np.arange(0, 2_000_000, 4000, dtype=np.int64)
    assert coverage(regular, 0, 2_000_000, 4000) == (1.0, 0.0, 2.0)
    with_hole = regular[(regular < 800_000) | (regular >= 1_000_000)]
    fraction, gap, _ = coverage(with_hole, 0, 2_000_000, 4000)
    assert np.isclose(fraction, .90) and gap == 200
    assert gap > config["max_uncovered_ms"]
    assert coverage(np.array([], dtype=np.int64), 0, 2_000_000, 4000) == (0.0, 2000.0, 0.0)
    # Half-open boundary: the sample at exactly 2 s belongs to the next window.
    boundary = np.arange(0, 4_000_000, 4000)
    left, right = np.searchsorted(boundary, [0, 2_000_000])
    assert right-left == 500 and boundary[right] == 2_000_000
    # Count a 100 ms empty leading edge even when internal samples are dense:
    # it counts against coverage and is not compressed out of clock time.
    fraction, gap, _ = coverage(regular[25:], 0, 2_000_000, 4000)
    assert np.isclose(fraction, .95) and gap == 100
    return {"status": "PASS", "protected_a1_files": len(provenance["protected_a1_sha256"]),
            "development_participants": sorted(development, key=int),
            "locked_test_participants": sorted(locked, key=int),
            "checked_windows": len(windows), "checked_candidates": len(audit),
            "checks": ["A1 byte hashes unchanged", "source/config/window provenance",
                       "participant separation", "explicit numeric feature allowlists",
                       "raw screen boundaries and half-open non-overlapping windows",
                       "coverage and maximum uncovered interval", "synthetic holes and boundary cases"],
            "test_model_evaluation": "not performed"}


if __name__ == "__main__":
    report = run_checks()
    write_json(A3 / "results/data/checks.json", report)
    print(json.dumps(report, indent=2))
