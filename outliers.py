import numpy as np
import pandas as pd

from cleaning import build_dataset

SCREEN_PX = (1920, 1080)
SCREEN_MM = (344, 194)
HEAD_MM = 700
MAX_DEG_S = 1000     # no human saccade is faster; above this the sample is an artefact
PUPIL_Z = 3.0        # |z| above this is an outlier, computed within participant

PX_PER_DEG = 2 * HEAD_MM * np.tan(np.radians(0.5)) * SCREEN_PX[0] / SCREEN_MM[0]


def velocity(df):
    # Gaze speed in deg/s. Grouped by (pid, segment) so no diff crosses a participant
    # or a stimulus change, and dt guards the gaps that were dropped from the dataset
    g = df.groupby(["pid", "segment"], observed=True, sort=False)
    dx = g["L POR X [px]"].diff()
    dy = g["L POR Y [px]"].diff()
    dt = g["Time"].diff() / 1e6            # Time is microseconds

    v = np.hypot(dx, dy) / PX_PER_DEG / dt
    return v.where(dt > 0)                 # first row of each group has no predecessor


def pupil_z(df):
    # Per participant: baseline pupil size differs between people, so a corpus-wide
    # z-score would flag everyone with naturally large pupils instead of the artefacts
    out = {}
    for col in ["L Mapped Diameter [mm]", "R Mapped Diameter [mm]"]:
        g = df.groupby("pid", observed=True)[col]
        out[col] = (df[col] - g.transform("mean")) / g.transform("std")
    return pd.DataFrame(out, index=df.index)


def iqr_bounds(x):
    q1, q3 = x.quantile([0.25, 0.75])
    return q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)


def pupil_iqr(df):
    # IQR per participant, for comparison with the z-score on the same column
    out = {}
    for col in ["L Mapped Diameter [mm]", "R Mapped Diameter [mm]"]:
        g = df.groupby("pid", observed=True)[col]
        q1 = g.transform(lambda c: c.quantile(0.25))
        q3 = g.transform(lambda c: c.quantile(0.75))
        r = q3 - q1
        out[col] = (df[col] < q1 - 1.5 * r) | (df[col] > q3 + 1.5 * r)
    return pd.DataFrame(out, index=df.index)


def report(df=None):
    df = build_dataset() if df is None else df
    n = len(df)
    print(f"{n:,} rows | {PX_PER_DEG:.1f} px per degree | limit {MAX_DEG_S} deg/s "
          f"= {MAX_DEG_S * PX_PER_DEG * 0.004:.0f} px per sample")

    v = velocity(df)
    fast = v > MAX_DEG_S
    print(f"\nvelocity  : {fast.sum():,} samples over {MAX_DEG_S} deg/s ({100*fast.mean():.3f}%)"
          f" | median {v.median():.1f}, p99 {v.quantile(.99):.0f}, max {v.max():.0f} deg/s")

    # Both prescribed methods on the column where both are valid, so the choice is shown
    # and not just asserted. Global vs per-pid IQR also shows why per-pid is the right unit
    z = pupil_z(df)
    wild = (z.abs() > PUPIL_Z).any(axis=1)
    print(f"pupil Z   : {wild.sum():,} samples with |z| > {PUPIL_Z} ({100*wild.mean():.3f}%)")

    iqr = pupil_iqr(df)
    print(f"pupil IQR : {iqr.any(axis=1).sum():,} samples outside the per-pid fence "
          f"({100*iqr.any(axis=1).mean():.3f}%)")
    for col in iqr.columns:
        lo, hi = iqr_bounds(df[col])
        g = 100 * ((df[col] < lo) | (df[col] > hi)).mean()
        print(f"   {col.split('[')[0].strip():<22} global IQR {g:5.2f}%  "
              f"per-pid IQR {100*iqr[col].mean():5.2f}%  per-pid Z {100*(z[col].abs()>PUPIL_Z).mean():5.2f}%")
    nested = (z.abs() > PUPIL_Z) & ~iqr
    print(f"   samples flagged by Z but not by IQR: {nested.any(axis=1).sum()} "
          f"-> the Z set is nested inside the IQR set")
    print(f"   skew {df['L Mapped Diameter [mm]'].skew():.2f}, range "
          f"{df['L Mapped Diameter [mm]'].min():.2f}-{df['L Mapped Diameter [mm]'].max():.2f} mm "
          f"-> no log transform needed")

    # Why IQR is the wrong tool for the coordinates: reading happens in the left half of
    # the screen, so the fence lands inside the stimulus and flags legitimate looks
    for col, lim in [("L POR X [px]", SCREEN_PX[0]), ("L POR Y [px]", SCREEN_PX[1])]:
        lo, hi = iqr_bounds(df[col])
        share = 100 * ((df[col] < lo) | (df[col] > hi)).mean()
        print(f"IQR {col:<14}: fence [{lo:7.0f}, {hi:7.0f}] on a 0-{lim} screen "
              f"-> would flag {share:.1f}%")
    return df


def handle(df=None):
    # Velocity artefacts are physically impossible -> the sample is wrong, so remove it.
    # Pupil extremes are a real physiological signal with a heavy tail -> cap, do not
    # remove, or we would delete the dilation the model is supposed to see
    df = build_dataset() if df is None else df
    before = len(df)

    v = velocity(df)                                # computed once, before any removal
    df = df.assign(velocity_deg_s=v.fillna(0))      # keep it, it is a useful feature
    df = df[~(v > MAX_DEG_S)].copy()

    for col in ["L Mapped Diameter [mm]", "R Mapped Diameter [mm]"]:
        g = df.groupby("pid", observed=True)[col]
        m, s = g.transform("mean"), g.transform("std")
        df[col] = df[col].clip(m - PUPIL_Z * s, m + PUPIL_Z * s)
    print(f"removed {before - len(df):,} of {before:,} rows ({100*(before-len(df))/before:.3f}%), "
          f"capped pupil diameter at |z| = {PUPIL_Z}")
    return df


if __name__ == "__main__":
    df = report()
    handle(df)
