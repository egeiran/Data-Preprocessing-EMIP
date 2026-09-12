import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from outliers import handle

TEST_SIZE = 0.2
SEED = 42

# Dropped, with the reason, because "why not a feature" is as much of a decision as the rest
DROPPED = {
    "pid": "the split key - as a feature it is participant identity",
    "Time": "an absolute timestamp, carries no gaze information",
    "Type": "constant after cleaning (SMP only) - zero variance",
    "Trial": "constant (1) in all 33 files",
    "segment": "session position, and the stimulus order is 61/39 -> leaks the target by design",
    "stimulus": "the target is derived from it",
    "R POR X [px]": "byte-identical to L POR X on 100% of valid rows",
    "R POR Y [px]": "byte-identical to L POR Y on 100% of valid rows",
}

# The corneal reflections are intermediate sensor readings, correlated with Raw at r ~ 1.0,
# and they hold the only sentinel zeros left after cleaning (up to 0.46% of rows). POR is
# derived from them, so their information is already in the data
CR_COLS = [f"{e} CR{i} {a} [px]" for e in "LR" for i in (1, 2) for a in "XY"]

ORDINAL = ["Pupil Confidence"]          # 0/1/2, ordered -> keep as one integer column
BINARY = ["quality"]                    # ok / interpolated
ONEHOT = ["screen_type", "language"]    # unordered -> one column per level


def label(df):
    # Both the target and two categorical features come out of the stimulus name
    s = df["stimulus"].astype(str)
    df = df[s.str.contains("vehicle|rectangle")].copy()
    s = df["stimulus"].astype(str)

    df["program"] = np.where(s.str.contains("vehicle"), "vehicle", "rectangle")
    df["screen_type"] = np.where(s.str.startswith("mupliple"), "choice", "code")
    # Language is a property of the participant, not of the screen: the multiple-choice
    # screens do not carry it in their filename, so take it from that participant's code screen
    code = s.str.endswith(("java.jpg", "java2.jpg", "scala.jpg"))
    lang = s.where(code).str.replace(".jpg", "", regex=False).str.split("_").str[1]
    df["language"] = df["pid"].map(lang.groupby(df["pid"], observed=True).first())

    # 22 and 39 kept their choice screens but lost both code screens to the coverage rule,
    # so their language is unknown. Requiring one makes the rule consistent: a participant
    # has to have contributed reading data to be in the modelling table at all
    unknown = df["language"].isna()
    if unknown.any():
        print(f"dropping {unknown.sum():,} rows from {df.loc[unknown, 'pid'].nunique()} "
              f"participants with no surviving code screen")
    return df[~unknown].copy()


def encode(df):
    y = (df["program"] == "vehicle").astype(int)      # label encoding: binary target
    groups = df["pid"].astype(str)

    X = df.drop(columns=list(DROPPED) + CR_COLS + ["program"], errors="ignore")
    X["quality"] = (X["quality"].astype(str) == "interpolated").astype(int)
    X["Pupil Confidence"] = X["Pupil Confidence"].astype(int)
    # drop_first avoids the dummy-variable trap: the full set is perfectly collinear,
    # which matters for PCA and for any linear model
    X = pd.get_dummies(X, columns=ONEHOT, drop_first=True, dtype=np.uint8)

    continuous = [c for c in X.columns
                  if c not in ORDINAL + BINARY and not c.startswith(tuple(ONEHOT))]
    return X, y, groups, continuous


def split(X, y, groups):
    # Grouped on participant: adjacent samples are 4 ms apart and near-identical, so a
    # random row split would put copies of the same moment in both sets
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    tr, te = next(gss.split(X, y, groups))
    return X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te], groups.iloc[tr], groups.iloc[te]


def scale(Xtr, Xte, continuous):
    # Fit on train only. Fitting on everything would put the test set's mean and sd into
    # the training data. One-hot and ordinal columns are left alone on purpose
    sc = StandardScaler().fit(Xtr[continuous])
    Xtr, Xte = Xtr.copy(), Xte.copy()
    Xtr[continuous] = sc.transform(Xtr[continuous]).astype(np.float32)
    Xte[continuous] = sc.transform(Xte[continuous]).astype(np.float32)
    return Xtr, Xte, sc


def report():
    df = label(handle())
    X, y, groups, continuous = encode(df)

    print(f"\n=== 4a. Encoding ===")
    print(f"target      program -> label encoded, {y.mean():.1%} vehicle")
    print(f"ordinal     {ORDINAL} kept as integers (0 < 1 < 2)")
    print(f"binary      {BINARY} -> 0/1")
    for c in ONEHOT:
        print(f"one-hot     {c} -> {sorted(df[c].unique())}")
    print(f"dropped     {len(DROPPED)} named + {len(CR_COLS)} CR columns; "
          f"{X.shape[1]} features remain ({len(continuous)} continuous)")

    print(f"\n=== 4b. Scaling: why it is needed ===")
    r = X[continuous].agg(["min", "max"]).T
    r["range"] = r["max"] - r["min"]
    print(r.sort_values("range", ascending=False).head(4).round(2).to_string())
    print(r.sort_values("range").head(3).round(3).to_string())

    Xtr, Xte, ytr, yte, gtr, gte = split(X, y, groups)
    Xtr, Xte, sc = scale(Xtr, Xte, continuous)

    print(f"\n=== 5. Split (grouped on participant) ===")
    print(f"train {len(Xtr):>9,} rows / {gtr.nunique():>2} participants / {ytr.mean():.1%} vehicle")
    print(f"test  {len(Xte):>9,} rows / {gte.nunique():>2} participants / {yte.mean():.1%} vehicle")
    print(f"overlap in participants: {len(set(gtr) & set(gte))}")
    for c in [c for c in Xtr.columns if c.startswith("language_")]:
        if Xtr[c].sum() == 0 or Xte[c].sum() == 0:
            print(f"  NB {c} is all-zero on one side: only 1 participant saw that version, "
                  f"and a group split cannot put them in both sets")
    print(f"after scaling, train continuous mean {Xtr[continuous].to_numpy().mean():+.2e}, "
          f"sd {Xtr[continuous].to_numpy().std():.3f}")
    print(f"test  continuous mean {Xte[continuous].to_numpy().mean():+.3f} "
          f"(not exactly 0 - the scaler never saw it, which is the point)")

    print(f"\n=== 6. PCA (bonus) ===")
    pca = PCA().fit(Xtr[continuous])
    ev = pca.explained_variance_ratio_.cumsum()
    for t in (0.90, 0.95, 0.99):
        print(f"  {int(t*100)}% of variance: {int(np.argmax(ev >= t)) + 1} of {len(continuous)} components")
    print(f"  first component alone: {pca.explained_variance_ratio_[0]:.1%}")
    return Xtr, Xte, ytr, yte


if __name__ == "__main__":
    report()
