import pyarrow.parquet as pq
import pandas as pd

import uproot

import numpy as np


def decode_keys(d):
    for k in list(d.keys()):
        d[k.decode("utf-8")] = d.pop(k)
    return d


def print_df_repeated_header(df, n_repeat_header=50):
    index_width = max([len(b) for b in df.index])
    padded_index = [b + " " * (index_width - len(b)) for b in df.index]
    n_repeat_header = 30
    i = 0
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        while i < len(df):
            df_slice = df[i : i + n_repeat_header]
            df_slice.index = padded_index[i : i + n_repeat_header]
            print()
            print(df_slice)
            i += n_repeat_header


dataset = pq.ParquetDataset("/home/llr/cms/rembser/WWZ-leptonic-EFT-analysis/skims/2017_WWZ_for_jetmet_sync")

nano = dataset.read_pandas().to_pandas()

baby = uproot.open("/home/llr/cms/rembser/scratch/baby-ntuples/WVZMVA2017_v0.1.21/wwz_4l2v_amcatnlo_1.root")["t"]
baby_event = baby.array("evt")

# print(baby.array("evt_scale1fb")[0])

all_branches = list(set([br.decode("utf-8") for br in baby.keys()]))


nano_event = nano["evt"]

n_overlap = np.sum(np.in1d(nano_event, baby_event))
n_nano = len(nano_event)
n_baby = len(baby_event)

print()
print("Overlapping events:", n_overlap)
print("Events only in nano:", n_nano - n_overlap)
print("Events only in baby:", n_baby - n_overlap)
print()

_, nano_idx, baby_idx = np.intersect1d(nano_event, baby_event, return_indices=True)

df_info = pd.DataFrame(index=sorted(all_branches))

for branch in all_branches:
    if not branch in nano.columns:
        continue

    # tgt = np.concatenate([converters[branch](nano)[nano_idx] for nano, nano_idx in zip(nanos, nano_indices)])
    # nano_index = b
    tgt = nano[branch][nano_idx]
    ref = baby.array(branch)[baby_idx]

    df_info.loc[branch, "exact match [%]"] = (tgt == ref).sum() * 100.0 / len(tgt)
    df_info.loc[branch, "float match [%]"] = np.abs(((tgt - ref) / ref) < 1e-6).sum() * 100.0 / len(tgt)
    df_info.loc[branch, "bias [tgt - ref]"] = np.mean(tgt - ref)

# print_df_repeated_header(df_info)

print(df_info.dropna())
# print_df_repeated_header(df_info)
# df_info.to_html("summary.html")
