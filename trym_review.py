# Independent validation of Tasks 4–6 created by Trym.
# This is not part of, or a replacement for, the original implementation.
# The script is more complex than a standard validation because I had too little
# RAM in WSL (about 4 GB) to run the entire process involving large data copies.
# I therefore used AI to help me devise a solution that uses less memory:
# processing one column at a time and performing PCA calculations in small batches,
# using the same method choices as in the original implementation.
"""Read the existing cache only; print results without writing data or figures.

Reuses the project's labelling, encoding, velocity and split definitions. Pupil
capping still uses ALL velocity-retained rows per participant BEFORE labelling,
exactly as outliers.handle does (it is not changed to training-only capping).
The cache is assumed to come from cleaning.py; cleaning itself is not validated.

StandardScaler is fitted one column at a time, which is equivalent to its
independent per-feature calculations. PCA explained variance is computed from
the full training covariance in small batches, without sampling or IncrementalPCA.
This validates the PCA variance spectrum, not sklearn's component vectors or
scores. Float64 covariance arithmetic can differ slightly from sklearn's chosen
solver; training values are cast to float32 just as in transform.scale.
"""

import sys
from pathlib import Path

# Also prevent imported project modules from creating __pycache__ files.
sys.dont_write_bytecode = True


def main():
    root = Path(__file__).resolve().parent
    # Check before third-party imports so a missing cache gives a useful message.
    if not (root / "dataset.parquet").is_file():
        raise SystemExit("Missing dataset.parquet: supply the existing cleaned cache. "
                         "This review will not create or rebuild it.")

    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler

    import cleaning
    import outliers as o
    import transform as t

    cache = root / cleaning.CACHE
    schema = pq.read_schema(cache).names
    metadata = ["pid", "stimulus", "quality", "Pupil Confidence"]
    velocity_cols = ["pid", "segment", "Time", "L POR X [px]", "L POR Y [px]"]
    print("Trym's independent review, existing cache, no file output", flush=True)
    narrow = pd.read_parquet(cache, columns=velocity_cols)
    velocity = o.velocity(narrow)
    keep = ~(velocity > o.MAX_DEG_S)
    # Capping groups include non-modelling screens, as in the original pipeline.
    retained_pid = narrow.loc[keep, "pid"]
    del narrow

    meta = pd.read_parquet(cache, columns=metadata)
    meta = t.label(meta.loc[keep])
    positions = meta.index.to_numpy(copy=True)
    encoded, y, groups, _ = t.encode(meta)
    del meta

    dropped = set(t.DROPPED) | set(t.CR_COLS)
    raw_features = [c for c in schema if c not in dropped]
    feature_names = [c for c in raw_features if c not in t.ONEHOT]
    feature_names += ["velocity_deg_s"]
    feature_names += [c for c in encoded.columns if c.startswith(tuple(t.ONEHOT))]
    continuous = [c for c in feature_names
                  if c not in t.ORDINAL + t.BINARY
                  and not c.startswith(tuple(t.ONEHOT))]

    if not encoded["quality"].isin([0, 1]).all():
        raise ValueError("Invalid binary quality encoding")
    if not encoded["Pupil Confidence"].isin([0, 1, 2]).all():
        raise ValueError("Unexpected ordinal Pupil Confidence value")
    dummy_cols = [c for c in encoded if c.startswith(tuple(t.ONEHOT))]
    if not encoded[dummy_cols].isin([0, 1]).all().all():
        raise ValueError("Invalid one-hot encoding")

    # Same row order, string participant keys, seed and split size as t.split.
    splitter = GroupShuffleSplit(n_splits=1, test_size=t.TEST_SIZE, random_state=t.SEED)
    train, test = next(splitter.split(encoded, y, groups))
    train_people = set(groups.iloc[train])
    test_people = set(groups.iloc[test])
    overlap = len(train_people & test_people)
    if overlap:
        raise ValueError("Participants overlap between training and test")
    print(f"Modelling rows: {len(y):,}")
    for value, name in [(0, "rectangle"), (1, "vehicle")]:
        count = int((y == value).sum())
        print(f"  {name} ({value}): {count:,} / {count / len(y):.2%}")
    print(f"Features: {len(feature_names)} total / {len(continuous)} continuous")
    print("Encoding: project label/encode reused; ordinal, binary and one-hot checks passed")
    print("One-hot columns: " + ", ".join(dummy_cols))
    print(f"Train: {len(train):,} rows / {len(train_people)} participants / "
          f"{y.iloc[train].mean():.1%} vehicle")
    print(f"Test:  {len(test):,} rows / {len(test_people)} participants / "
          f"{y.iloc[test].mean():.1%} vehicle")
    print(f"Participant overlap: {overlap}")
    del encoded, y, groups

    # Only one full numeric training matrix (~101 MiB for the report's data).
    # Read other numeric columns separately, retaining original float64 precision
    # through capping and scaling. No full modelling/test DataFrame is made.
    scaled_train = np.empty((len(train), len(continuous)), dtype=np.float32, order="F")
    means, stds, test_means = [], [], []
    expected_stds = []
    for j, col in enumerate(continuous):
        if col == "velocity_deg_s":
            values = velocity.fillna(0).iloc[positions].to_numpy()
        else:
            series = pd.read_parquet(cache, columns=[col])[col]
            if col in ["L Mapped Diameter [mm]", "R Mapped Diameter [mm]"]:
                series = series.loc[keep]
                g = series.groupby(retained_pid, observed=True)
                mean, std = g.transform("mean"), g.transform("std")
                series = series.clip(mean - o.PUPIL_Z * std, mean + o.PUPIL_Z * std)
                del g, mean, std
            values = series.loc[positions].to_numpy()
            del series
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite modelling values in {col}")
        train_col = values[train].reshape(-1, 1)
        scaler = StandardScaler().fit(train_col)  # TRAIN only, never test.
        scaled_train[:, j] = scaler.transform(train_col).ravel().astype(np.float32)
        test_col = scaler.transform(values[test].reshape(-1, 1)).astype(np.float32)
        means.append(scaled_train[:, j].mean(dtype=np.float64))
        stds.append(scaled_train[:, j].std(dtype=np.float64))
        expected_stds.append(0.0 if scaler.var_[0] == 0 else 1.0)
        test_means.append(test_col.mean(dtype=np.float64))
        del values, train_col, test_col, scaler
    del velocity, keep, retained_pid, positions

    scaling_ok = (np.allclose(means, 0, atol=1e-6)
                  and np.allclose(stds, expected_stds, atol=1e-6))
    print("Scaler fitted on TRAIN only: yes (independent per-column StandardScaler)")
    print(f"Scaled train: mean {np.mean(means):+.2e}, "
          f"per-column std range {min(stds):.6f}–{max(stds):.6f}; "
          f"check {'PASS' if scaling_ok else 'FAIL'}")
    print(f"Scaled test mean: {np.mean(test_means):+.3f}")
    if not scaling_ok:
        raise ValueError("Scaled training mean/std check failed")

    # Full-data covariance PCA, accumulated from TRAIN only. Centering uses the
    # actual float32 training means. Only a small float64 batch is allocated.
    covariance = np.zeros((len(continuous), len(continuous)), dtype=np.float64)
    for start in range(0, len(train), 8192):
        batch = scaled_train[start:start + 8192].astype(np.float64)
        batch -= means
        covariance += batch.T @ batch
    covariance /= len(train) - 1
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1].clip(min=0)
    if eigenvalues.sum() <= 0:
        raise ValueError("PCA requires nonzero training variance")
    cumulative = np.cumsum(eigenvalues / eigenvalues.sum())
    print("PCA fitted on TRAIN only: yes (full training covariance eigendecomposition)")
    for threshold in (0.90, 0.95, 0.99):
        count = int(np.searchsorted(cumulative, threshold)) + 1
        print(f"  {threshold:.0%} variance: {count} components "
              f"(actual {cumulative[count - 1]:.2%})")
    print(f"First component: {eigenvalues[0] / eigenvalues.sum():.1%}")


if __name__ == "__main__":
    main()
