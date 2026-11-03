"""Fixed-grid gaze sequences for the CNN: the V2 code windows, plus instruction and
answer screens for transfer pretraining. A step without a sample stays empty (mask 0),
so holes are never compressed into continuous time."""
import hashlib
import json
import re

import numpy as np
import pandas as pd

from data import A3, ROOT, CACHE_COLUMNS, CODE_PATTERN, coverage, read_config, sha256, write_json
from quality import time_parts
from fixations import KEY, load_v2
import cleaning
import outliers

PLAN = A3 / "deep_plan.json"
SOURCE_PATTERN = r"^(instruction_calibration|instruction_comprehension|mupliple_choice_(vehicle|rectangle))\.jpg$"
CHANNELS = ["x", "y", "pupil_left", "pupil_right", "measured_mask", "interpolated_flag"]
SIGNALS = ["L POR X [px]", "L POR Y [px]", "L Mapped Diameter [mm]", "R Mapped Diameter [mm]"]


def to_grid(part, start, steps, period_us):
    """Raw px/mm values on the nominal 4 ms grid; at most one sample per step.
    Floor, not rounding: half-open windows keep every index below `steps`."""
    index = (part.Time.to_numpy(dtype=np.int64) - start) // period_us
    assert (index >= 0).all() and (index < steps).all()
    first = np.r_[True, np.diff(index) > 0]
    index, part = index[first], part[first]
    seq = np.zeros((len(CHANNELS), steps), dtype=np.float32)
    for channel, column in enumerate(SIGNALS):
        seq[channel, index] = part[column].to_numpy()
    seq[4, index] = part.quality.to_numpy() == "ok"
    seq[5, index] = part.quality.to_numpy() == "interpolated"
    return seq, int((~first).sum())


def retained(df, pattern):
    """Same sample filter as V2: screen pattern, then the A1 >1,000 deg/s velocity rule."""
    df = df[df.stimulus.str.match(pattern)].copy()
    return df.loc[~(outliers.velocity(df) > outliers.MAX_DEG_S)]


def screen_bounds(people):
    """Screen start/end from raw .jpg messages, numbered exactly as data.py and A1."""
    rows = []
    for path in [ROOT / p for p in cleaning.all_paths() if (ROOT / p).is_file()]:
        pid = path.name.split("_")[0]
        if pid not in people:
            continue
        raw = cleaning.load(path)
        period_us = int(1e6 / raw.attrs["sample_rate"])
        text = raw["L Raw X [px]"].astype(str).str.replace("# Message: ", "", regex=False)
        messages = raw.loc[(raw.Type == "MSG") & text.str.endswith(".jpg"), ["Time"]].copy()
        messages["stimulus"] = text.loc[messages.index]
        messages["segment"] = np.arange(1, len(messages) + 1)
        final_end = int(raw.loc[raw.Type == "SMP", "Time"].max()) + period_us
        messages["end_us"] = np.r_[messages.Time.to_numpy()[1:], final_end]
        for msg in messages.itertuples():
            if re.fullmatch(SOURCE_PATTERN, msg.stimulus):
                rows.append({"pid": pid, "segment": msg.segment, "stimulus": msg.stimulus,
                             "start_us": int(msg.Time), "end_us": int(msg.end_us), "period_us": period_us})
    return pd.DataFrame(rows)


def source_windows(groups, bounds, config, policy, steps):
    """Same window rules as the code screens: V1 coverage/gap and V2 measured share."""
    window_us = int(config["window_seconds"] * 1e6)
    sequences, rows, rejected, collisions = [], [], 0, 0
    for screen in bounds.itertuples():
        samples = groups.get((screen.pid, screen.segment))
        if samples is None:
            continue
        assert (samples.stimulus == screen.stimulus).all()
        times = samples.Time.to_numpy(dtype=np.int64)
        assert times.min() >= screen.start_us and times.max() < screen.end_us
        for index in range((screen.end_us - screen.start_us) // window_us):
            left, right = screen.start_us + index * window_us, screen.start_us + (index + 1) * window_us
            lo, hi = np.searchsorted(times, [left, right])
            fraction, gap_ms, _ = coverage(times[lo:hi], left, right, screen.period_us)
            measured, _, _ = time_parts(times[lo:hi], samples.quality.to_numpy()[lo:hi], left, right, screen.period_us)
            if (fraction < config["min_coverage"] or gap_ms > config["max_uncovered_ms"]
                    or measured / config["window_seconds"] < policy["min_measured_fraction"]):
                rejected += 1
                continue
            seq, dropped = to_grid(samples.iloc[lo:hi], left, steps, screen.period_us)
            assert int((seq[4] + seq[5]).sum()) + dropped == hi - lo
            collisions += dropped
            sequences.append(seq)
            rows.append({"pid": screen.pid, "segment": screen.segment, "window_index": index,
                         "stimulus": screen.stimulus, "source_target": int(screen.stimulus.startswith("mupliple_choice"))})
    return np.stack(sequences), pd.DataFrame(rows), rejected, collisions


def digest(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def main():
    plan = json.loads(PLAN.read_text())
    steps = plan["sequence"]["steps"]
    assert plan["sequence"]["channels"] == CHANNELS
    config = read_config()
    selection = json.loads((A3 / "results/quality/selection.json").read_text())
    v2 = load_v2(selection).sort_values(KEY).reset_index(drop=True)
    df = pd.read_parquet(ROOT / config["a1_cache"], columns=CACHE_COLUMNS)
    for col in ["pid", "stimulus", "quality"]:
        df[col] = df[col].astype(str)
    code = dict(tuple(retained(df, CODE_PATTERN).groupby(["pid", "segment"], sort=False)))
    code_sequences, code_collisions = [], 0
    for window in v2.itertuples():
        samples = code[(window.pid, window.segment)]
        left, right = np.searchsorted(samples.Time.to_numpy(), [window.start_us, window.end_us])
        seq, collisions = to_grid(samples.iloc[left:right], window.start_us, steps, 4000)
        assert int((seq[4] + seq[5]).sum()) + collisions == window.n_samples
        code_sequences.append(seq)
        code_collisions += collisions
    code_X = np.stack(code_sequences)
    # Transfer source: development people only; cnn.py narrows to each partition's training people.
    development = set(v2.loc[v2.split == "development", "pid"])
    locked = set(selection["policy"]["test_participants"])
    bounds = screen_bounds(development)
    source = dict(tuple(retained(df, SOURCE_PATTERN).groupby(["pid", "segment"], sort=False)))
    source_X, source_rows, rejected, source_collisions = source_windows(source, bounds, config, selection["policy"], steps)
    assert not set(source_rows.pid) & locked and set(source_rows.pid) <= development
    output = A3 / "cache"
    np.savez_compressed(output / "sequences.npz", code_X=code_X, code_pid=v2.pid.to_numpy(dtype=str),
                        code_segment=v2.segment.to_numpy(), code_window_index=v2.window_index.to_numpy(),
                        code_target=v2.target.to_numpy(), code_split=v2.split.to_numpy(dtype=str),
                        source_X=source_X, source_pid=source_rows.pid.to_numpy(dtype=str),
                        source_target=source_rows.source_target.to_numpy())
    evidence = {"channels": CHANNELS, "steps": steps,
                "code_windows": {s: int((v2.split == s).sum()) for s in ["development", "locked_test"]},
                "code_grid_collisions": code_collisions,
                "code_present_step_fraction_median": float(np.median((code_X[:, 4] + code_X[:, 5]).mean(axis=1))),
                "source_windows": len(source_rows), "source_rejected_windows": rejected,
                "source_grid_collisions": source_collisions,
                "source_people": int(source_rows.pid.nunique()),
                "source_class_windows": {"instruction": int((source_rows.source_target == 0).sum()),
                                         "answer_screen": int((source_rows.source_target == 1).sum())},
                "source_stimulus_windows": source_rows.stimulus.value_counts().to_dict(),
                "code_sha256": digest(code_X, v2.target.to_numpy()), "source_sha256": digest(source_X, source_rows.source_target.to_numpy()),
                "builder_sha256": sha256(__file__), "plan_sha256": sha256(PLAN),
                "checks": ["V2 keys against committed evidence", "every code sample on its own grid step or counted",
                           "every source sample on its own grid step or counted", "source screens match cache stimulus",
                           "source windows pass V1 and V2 quality rules", "no locked-test person in the source set"]}
    write_json(A3 / "results/deep_v3/sequences.json", evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
