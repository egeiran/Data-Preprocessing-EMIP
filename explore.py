"""Task 1, data exploration. Prints every number quoted in section 1 of report.md."""
import numpy as np
import pandas as pd

import cleaning

SHOW = ["Time", "Type", "L POR X [px]", "L POR Y [px]", "L Validity"]
STATS = ["L POR X [px]", "L POR Y [px]", "L Mapped Diameter [mm]"]
CATEGORICAL = ["Type", "L Validity", "R Validity", "Pupil Confidence"]


def one_file(path="emip_dataset/rawdata/2_rawdata.tsv"):
    raw = cleaning.load(path)
    smp = raw[raw["Type"] == "SMP"].copy()
    num = smp.columns.drop("Type")

    print(f"=== {path}: shape {raw.shape}, sample rate {raw.attrs['sample_rate']} Hz ===")
    print("\n--- first rows (selected columns): all zeros until the calibration MSG ---")
    first_msg = int(np.argmax((raw["Type"] == "MSG").to_numpy()))
    print(raw[SHOW].iloc[[0, 1, first_msg, first_msg + 1]].to_string())
    print(f"(first MSG row is row {first_msg + 1} of the file)")

    print("\n--- dtypes ---")
    print("whole file       :", raw.dtypes.astype(str).value_counts().to_dict(),
          "| L Raw X [px] is", raw["L Raw X [px]"].dtype, "because MSG text lives in it")
    smp[num] = smp[num].apply(pd.to_numeric, errors="coerce")
    print("SMP rows, numeric:", smp.dtypes.astype(str).value_counts().to_dict(),
          "| int64 column(s):", list(smp.dtypes[smp.dtypes == "int64"].index))

    nun = smp.nunique()
    print("\n--- columns with one unique value or none ---")
    print(list(nun[nun <= 1].index))

    print("\n--- unique values in categorical columns ---")
    for c in CATEGORICAL:
        vals = raw[c] if c == "Type" else smp[c]
        print(f"  {c:<17} {sorted(vals.dropna().unique())}")
    msgs = raw.loc[raw["Type"] == "MSG", "L Raw X [px]"].str.replace("# Message: ", "", regex=False)
    print("  messages         ", msgs.unique().tolist())

    clean, _ = cleaning.preprocess(raw)
    clean = clean[clean["quality"] != "gap"]
    print(f"\n--- summary statistics after cleaning (n = {len(clean)}) ---")
    d = clean[STATS].describe().T[["mean", "std", "min", "50%", "max"]]
    print(d.rename(columns={"50%": "median"}).round(2).to_string())


def corpus():
    n_smp = n_msg = 0
    rates, max_gap_ms, same_por = set(), 0.0, []
    quality = pd.Series(dtype=float)
    for path in cleaning.all_paths():
        try:
            raw = cleaning.load(path)
        except FileNotFoundError:
            continue
        n_smp += int((raw["Type"] == "SMP").sum())
        n_msg += int((raw["Type"] == "MSG").sum())
        rates.add(raw.attrs["sample_rate"])

        smp, _ = cleaning.preprocess(raw)
        quality = quality.add(smp["quality"].value_counts(), fill_value=0)
        gap = smp.groupby("segment")["Time"].diff().max() / 1000       # Time is microseconds
        max_gap_ms = max(max_gap_ms, float(gap))
        ok = smp[smp["quality"] == "ok"]
        same_por.append(float(((ok["L POR X [px]"] == ok["R POR X [px]"])
                               & (ok["L POR Y [px]"] == ok["R POR Y [px]"])).mean()))

    print(f"\n=== corpus: {n_smp:,} SMP rows and {n_msg} MSG rows across "
          f"{len(same_por)} files, sample rate(s) {sorted(rates)} Hz, "
          f"max within-segment gap {max_gap_ms:.0f} ms ===")
    q = 100 * quality / quality.sum()
    print("quality of raw samples:", q.round(1).to_dict(),
          f"-> {q['gap'] + q['interpolated']:.0f}% invalid")
    print(f"L POR == R POR on valid rows: min over files {100 * min(same_por):.1f}%")

    df = cleaning.build_dataset()
    num = df.select_dtypes("number")
    num = num.loc[:, num.nunique() > 1]
    corr = num.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack()
    print(f"\ncolumn pairs with |r| > 0.95 in the cleaned dataset: {(pairs > 0.95).sum()} "
          f"of {len(pairs)}")
    families = sorted({c.split(" ")[1] for c in num.columns if c.startswith(("L ", "R "))})
    print("gaze representations:", families)


if __name__ == "__main__":
    one_file()
    corpus()
