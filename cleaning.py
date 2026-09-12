import pathlib
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

SHORT_MS = 100     # holes shorter than this are interpolated; longer ones stay holes
DROPOUT_MS = 50    # under this a hole is sensor noise, not a blink (79% of holes are <20 ms)
BLINK_MAX_MS = 400 # blink duration is roughly 50-400 ms; above that it is track loss
WINDOW_S = 5       # trailing window for the gap features
MIN_CODE_S = 20    # a (participant, stimulus) cell needs this much usable data
CACHE = "dataset.parquet"

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

    # Trial is 1 everywhere, so MSG is the only real boundary -> no holes across stimuli.
    # Only the .jpg messages change the screen; a mouseclick is an event on the same screen
    text = data["L Raw X [px]"].astype(str).str.replace("# Message: ", "", regex=False)
    is_stim = (data["Type"] == "MSG") & text.str.endswith(".jpg")
    seg = is_stim.cumsum()
    smp["segment"] = seg

    lab = text[is_stim]
    lab.index = seg[is_stim]
    smp["stimulus"] = smp["segment"].map(lab).fillna("none")   # 0 = before the first stimulus

    missing = isInValid(smp)
    new_run = (missing != missing.shift(fill_value=False)) | (smp["segment"] != smp["segment"].shift())
    group = new_run.cumsum()

    holes = (pd.DataFrame({"group": group, "missing": missing})
            .query("missing")
            .groupby("group")
            .size()
            .reset_index(name="samples"))
    holes["ms"] = holes["samples"] * 1000 / rate
    holes["type"] = np.select([holes["ms"] < DROPOUT_MS, holes["ms"] < BLINK_MAX_MS],
                              ["dropout", "blink"], default="loss")

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
    meta_cols = ["Time", "Type", "Trial", "L Validity", "R Validity", "Pupil Confidence",
                 "segment", "stimulus"]
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

    # The gaps are behaviour, not just noise -> keep what they say before the rows are dropped.
    # Trailing window, not per stimulus: a per-stimulus mean would encode the label and leak
    blink = missing & (hole_ms >= DROPOUT_MS) & (hole_ms < BLINK_MAX_MS)
    smp["_onset"] = (blink & ~blink.shift(fill_value=False)).astype(float)
    smp["_gap"] = (smp["quality"] == "gap").astype(float)

    win = int(WINDOW_S * rate)
    r = smp.groupby("segment", sort=False)
    smp["blink_rate"] = r["_onset"].transform(lambda c: c.rolling(win, min_periods=1).sum()) / WINDOW_S
    smp["gap_fraction"] = r["_gap"].transform(lambda c: c.rolling(win, min_periods=1).mean())
    smp = smp.drop(columns=["_onset", "_gap"])

    smp.attrs["sample_rate"] = rate

    if verbose:
        print(ut)
        print(smp["quality"].value_counts(normalize=True).round(4).to_string())

    return smp, ut

def all_paths():
    paths = [f"emip_dataset/rawdata/{i}_rawdata.tsv" for i in range(1, 42)]
    return paths + ["emip_dataset/rawdata/211_rawdata.xlsx"]   # 211 is Excel, not tsv

def base_stimulus(stim):
    return stim.str.split("_").str[0]   # vehicle_java2.jpg -> vehicle

def code_seconds(smp):
    # usable seconds per code-reading stimulus, for one participant
    ok = smp[(smp["quality"] != "gap") & smp["stimulus"].str.startswith(("vehicle", "rectangle"))]
    return ok.groupby(base_stimulus(ok["stimulus"])).size() / smp.attrs["sample_rate"]

def each_participant():
    # one parse per file, shared by coverage() and build_dataset()
    for path in all_paths():
        try:
            smp, _ = preprocess(load(path))
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"{path}: {type(e).__name__}: {e}")
            continue
        yield path.split("/")[-1].split("_")[0], smp

def coverage(min_s=MIN_CODE_S):
    # gap_% measures the session, incl. breaks and instructions. What we model is code
    # reading, so exclude per (participant, stimulus) on usable seconds instead
    rows = [{"pid": pid, "stimulus": stim, "usable_s": round(sec, 1)}
            for pid, smp in each_participant()
            for stim, sec in code_seconds(smp).items()]

    t = pd.DataFrame(rows)
    t["keep"] = t["usable_s"] >= min_s
    print(t[~t["keep"]].to_string(index=False))
    print(f"\ndropped {(~t['keep']).sum()} of {len(t)} cells at >= {min_s} s")
    return t

def build_dataset(cache=CACHE, rebuild=False, min_s=MIN_CODE_S):
    if not rebuild and pathlib.Path(cache).exists():
        return pd.read_parquet(cache)

    frames = []
    for pid, smp in each_participant():
        sec = code_seconds(smp)
        thin = set(sec[sec < min_s].index)                 # e.g. 18 keeps vehicle, loses rectangle
        smp = smp[~base_stimulus(smp["stimulus"]).isin(thin)]

        smp = smp[smp["quality"] != "gap"]                 # coordinates here are unrecoverable
        smp.insert(0, "pid", pid)
        frames.append(smp)

    out = pd.concat(frames, ignore_index=True)
    for c in ["pid", "Type", "stimulus", "quality"]:
        out[c] = out[c].astype("category")

    out.to_parquet(cache, index=False)
    print(f"{len(out)} rows x {out.shape[1]} cols from {out['pid'].nunique()} participants -> {cache}")
    return out

def test_outliersets():
    rows = []
    for path in all_paths():
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

if __name__ == "__main__":
    data = load("emip_dataset/rawdata/2_rawdata.tsv")
    preprocess(data, 1)
    test_outliersets()
    coverage()
    build_dataset(rebuild=True)