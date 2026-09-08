import pandas as pd
import numpy as np

def load(path):
    with open(path) as fh:
        for i, line in enumerate(fh):
            if line.startswith("Time\t"):
                break
    df = pd.read_csv(path, sep="\t", skiprows=i, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df

def one_file(filename, verbose=0):
    df = load(f"emip_dataset/rawdata/{filename}")

    if verbose:
        print(df.head())

    smp = df[df["Type"] == "SMP"]       # Målinger 
    msg = df[df["Type"] == "MSG"]       # Meldinger

    ls = len(smp)
    lm = len(msg)

    smp_ = smp.copy()
    smp_ = smp_.drop(columns="Type").apply(pd.to_numeric, errors="coerce")
    smp_types = smp_.dtypes
    n = smp_.nunique()

    d = smp_.describe().T[["mean", "std", "min", "max"]]
    d.insert(0, "unique", n)
    d.insert(1, "share_is_0", (smp_ == 0).mean().round(3))
    d.insert(0, "dtype", smp_types.astype(str))

    if verbose:  
        print()
        print("=== Measurements ===")
        print("Description:", d, sep="\n")

    # Validity
    l_valids = (smp_["L Validity"] == 0).sum()
    r_valids = (smp_["R Validity"] == 0).sum()

    l_valid_zeroes = ((smp_["L POR X [px]"] == 0) & (smp_["L Validity"] == 1)).sum()
    l_zeroes = (smp_["L POR X [px]"] == 0).sum()

    r_valid_zeroes = ((smp_["R POR X [px]"] == 0) & (smp_["R Validity"] == 1)).sum()
    r_zeroes = (smp_["R POR X [px]"] == 0).sum()

    if verbose:
        print()
        print("== Left ==")
        print("Valid", l_valids)
        print(f"Valid zeroes (zeroes) {l_valid_zeroes} ({l_zeroes})")
        print("== Right ==")
        print("Valid", r_valids)
        print(f"Valid zeroes (zeroes) {r_valid_zeroes} ({r_zeroes})")

    # Outliers
    x, y = smp_["L POR X [px]"], smp_["L POR Y [px]"]
    off = ((x < 0) | (x > 1920) | (y < 0) | (y > 1080))

    missing = (smp_["L Validity"] == 0).values
    group = (missing != np.r_[False, missing[:-1]]).cumsum()
    holes = (pd.DataFrame({"group": group, "missing": missing})
            .query("missing")
            .groupby("group")
            .size()
            .reset_index(name="samples"))
    holes["ms"] = holes["samples"] * 1000 / 250   # Sjekk om 250 (hz) er bildefrekvensen til alle
    holes["type"] = np.where(holes["ms"] < 150, "blink", "loss")   # Er 150 valid her? TODO
    ut = holes[["samples", "ms", "type"]]

    ut = (ut.groupby("type")
            .agg(antall=("ms", "size"),
            median_ms=("ms", "size"),
            max_ms=("ms", "max"),
            lost_samples=("samples", "sum")))

    ut["share_of_recording"] = (ut["lost_samples"] / ls).sum()
    print(filename, (ut["lost_samples"] / ls).sum())


    if verbose or 1:
        print()
        print("== Outliers and faulty data ==")
        print("Off the screen", off.sum())
        print(ut)

    msg_ = msg.copy()
    df = msg_[["Time", "L Raw X [px]"]]
    # df.rename(columns={"L Raw X [px]": "Message"}, inplace=True) # TODO: Finn feilen her
    # df = df[["Time", "Message"]]
    df = df[["Time", "L Raw X [px]"]]

    if verbose:
        print()
        print("=== Messages ===")
        print(df)

    if verbose:
        print()
        print("=== Totals ===")
        print("Lines with measurements: ", ls)
        print("Lines with messages", lm)

    return df, smp, msg, ls, lm

def summarize():
    tot_ls = 0
    tot_lm = 0
    faulty = 0
    not_found = 0
    correct = 0
    # for i in range(1, 42):
    for i in [1, 9, 18, 20, 22, 23, 24, 39, 41]:
        
        try:
            filename = f"{i}_rawdata.tsv"
            _cdf, _csmp, _cmsg, cls, clm = one_file(filename)
            print()
            print("File", i)
            tot_ls += cls
            tot_lm += clm
            correct += 1
        except FileNotFoundError:
            not_found += 1
        except:
            print("Faulty set:", i)
            faulty += 1

    print("====== SUMMARY ======")
    print(f"Average amound of measurements: {tot_ls/correct}")
    print(f"Average amount of messages: {tot_lm/correct}")
    print(f"Faulty files (correct): {faulty} ({correct})")

one_file("10_rawdata.tsv", 1)
# summarize()
