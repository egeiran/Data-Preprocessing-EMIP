import pandas as pd
import numpy as np

def load(path):
    if str(path).endswith(".xlsx"):
        return load_xlsx(path)

    rate = None
    with open(path) as fh:
        for i, line in enumerate(fh):
            if line.startswith("## Sample Rate:"):
                rate = float(line.split(":", 1)[1])
            if line.startswith("Time\t"):
                break
    df = pd.read_csv(path, sep="\t", skiprows=i, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df.attrs["sample_rate"] = rate
    return df

def load_xlsx(path):   # 211 is the same format, just saved as Excel
    head = pd.read_excel(path, header=None, nrows=60)
    col0 = head[0].astype(str)

    rate = head.loc[col0.str.startswith("## Sample Rate:"), 1]
    rate = float(rate.iloc[0]) if len(rate) else None

    i = head.index[col0.str.startswith("Time")][0]
    df = pd.read_excel(path, skiprows=i)
    df.columns = [str(c).strip() for c in df.columns]
    df.attrs["sample_rate"] = rate
    return df

SHORT_MS = 100   # holes shorter than this are interpolated; longer ones stay holes

def preprocess(data_in, verbose=0):
    # Remove all columns with no data (either 1 or 0 unique values)
    data = data_in.drop(columns=["Aux1", "R Plane", "L Plane", "Timing", "Frame"])
    smp = data[data["Type"] == "SMP"].copy()   # Split to SMP
    msg = data[data["Type"] == "MSG"].copy()   # Split to MSG

    # MSG rows put text in "L Raw X [px]" -> object dtype. Everything in smp is numbers
    num = smp.columns.drop("Type")
    smp[num] = smp[num].apply(pd.to_numeric, errors="coerce")

    rate = data_in.attrs.get("sample_rate") or 250   # Hz, read from the file header
    n_smp = len(smp)   # number of measurements; MSG rows do not count

    # Fill holes with Holes instead of a lot of "0-rows"
    # TODO: Convert data["Type"] to 1 or 0 -> Encoding (maybe use holes here as well (Blink and loss))

    def isInValid(smp, screen=(1920, 1080)):
        l_bad = (smp["L Validity"] == 0)
        r_bad = (smp["R Validity"] == 0)
        # <= 0, not < 0: 0.0 is the "no data" sentinel (8223 samples are valid at (0,0))
        x_out = (smp["L POR X [px]"] <= 0) | (smp["L POR X [px]"] > screen[0]) \
              | (smp["R POR X [px]"] <= 0) | (smp["R POR X [px]"] > screen[0])
        y_out = (smp["L POR Y [px]"] <= 0) | (smp["L POR Y [px]"] > screen[1]) \
              | (smp["R POR Y [px]"] <= 0) | (smp["R POR Y [px]"] > screen[1])
        return l_bad | r_bad | x_out | y_out

    # Trial is 1 everywhere, so MSG is the only real boundary -> no holes across stimuli
    smp["segment"] = (data["Type"] == "MSG").cumsum()

    missing = isInValid(smp)
    new_run = (missing != missing.shift(fill_value=False)) | (smp["segment"] != smp["segment"].shift())
    group = new_run.cumsum()

    holes = (pd.DataFrame({"group": group, "missing": missing})
            .query("missing")
            .groupby("group")
            .size()
            .reset_index(name="samples"))
    holes["ms"] = holes["samples"] * 1000 / rate
    holes["type"] = np.where(holes["ms"] < 150, "blink", "loss")   # Er 150 valid her? TODO

    ut = holes[["samples", "ms", "type"]]

    ut = (ut.groupby("type")
            .agg(antall=("ms", "size"),
            median_ms=("ms", "median"),
            max_ms=("ms", "max"),
            lost_samples=("samples", "sum")))

    ut["share_of_recording"] = ut["lost_samples"] / n_smp

    # ---------- Interpolation ----------
    run_len = missing.groupby(group).transform("size")
    hole_ms = run_len * 1000 / rate   # length of the hole THIS row belongs to

    long_ = missing & (hole_ms > SHORT_MS)

    # Validity/Confidence describe the measurement, they are not it -> never interpolated
    meta_cols = ["Time", "Type", "Trial", "L Validity", "R Validity", "Pupil Confidence", "segment"]
    signal = [c for c in smp.columns if c not in meta_cols]

    smp.loc[missing, signal] = np.nan   # as numbers the 0-rows would pull the fill to (0,0)

    # Against Time, not row number. limit_area keeps the leading 0-rows unfilled
    sig = smp[signal].set_axis(smp["Time"].to_numpy())
    sig = sig.groupby(smp["segment"].to_numpy(), sort=False).transform(
        lambda g: g.interpolate(method="index", limit_area="inside"))
    smp[signal] = sig.to_numpy()

    # NB: interpolate(limit=n) fills the first n NaNs of ANY hole, so it cannot be used here
    smp.loc[long_, signal] = np.nan

    # From actual NaN, not from the length: a short hole at a segment edge has nothing to fill from
    smp["quality"] = np.where(~missing, "ok",
                     np.where(smp["L POR X [px]"].notna(), "interpolated", "gap"))

    if verbose:
        print(ut)
        print(smp["quality"].value_counts(normalize=True).round(4).to_string())

    return smp, ut

def test_outliersets():
    rows = []
    paths = [f"emip_dataset/rawdata/{i}_rawdata.tsv" for i in range(1, 42)]
    paths.append("emip_dataset/rawdata/211_rawdata.xlsx")

    for path in paths:
        try:
            smp, ut = preprocess(load(path))
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"{path}: {type(e).__name__}: {e}")   # ikke svelg feil stille
            continue

        q = smp["quality"].value_counts()
        n = len(smp)
        rows.append({
            "fil": path.split("/")[-1].split("_")[0],
            "samples": n,
            "ok_%": round(100 * q.get("ok", 0) / n, 1),
            "interp_%": round(100 * q.get("interpolated", 0) / n, 1),
            "gap_%": round(100 * q.get("gap", 0) / n, 1),
        })

    t = pd.DataFrame(rows).sort_values("gap_%", ascending=False)
    print(t.to_string(index=False))
    return t

data = load(f"emip_dataset/rawdata/2_rawdata.tsv")
preprocess(data, 1)
# test_outliersets()