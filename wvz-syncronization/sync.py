import uproot
import os
import numpy as np
import pandas as pd
import argparse


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


parser = argparse.ArgumentParser(description="WVZ syncronization effort.")
parser.add_argument("baby", type=str, help="the baby file")
parser.add_argument("nanos", type=str, nargs="+", help="the nano file(s)")
parser.add_argument("--data", action="store_true")
parser.add_argument("--lumi", type=float)

args = parser.parse_args()

baby = uproot.open(args.baby)["t"]
baby_event = baby.array("evt")

print(baby.array("evt_scale1fb")[0])

all_branches = list(set([br.decode("utf-8") for br in baby.keys()]))

nanos = [uproot.open(nano)["Events"] for nano in args.nanos]
nano_events = [nano.array("event") for nano in nanos]

n_overlap = np.sum(np.in1d(np.concatenate(nano_events), baby_event))
n_nano = np.sum([len(nano_event) for nano_event in nano_events])
n_baby = len(baby_event)

print()
print("Overlapping events:", n_overlap)
print("Events only in nano:", n_nano - n_overlap)
print("Events only in baby:", n_baby - n_overlap)
print()

nano_indices = []
baby_indices = []

for nano_event in nano_events:
    res = np.intersect1d(nano_event, baby_event, return_indices=True)
    nano_indices.append(res[1])
    baby_indices.append(res[2])

converters = {
    "run": lambda nano: nano.array("run"),
    "lumi": lambda nano: nano.array("luminosityBlock"),
    "evt": lambda nano: nano.array("event"),
    "isData": lambda nano: np.zeros(len(nano), dtype=np.bool) | args.data,
    "evt_passgoodrunlist": lambda nano: np.ones(len(nano), dtype=np.bool),
    "evt_firstgoodvertex": lambda nano: np.ones(len(nano), dtype=np.int),
    "nvtx": lambda nano: nano.array("PV_npvsGood"),
    "nTrueInt": lambda nano: np.zeros(len(nano), dtype=np.int) - 999 if args.data else nano.array("Pileup_nTrueInt"),
    "passesMETfiltersRun2": lambda nano: nano.array("Flag_METFilters"),
    "met_gen_pt": lambda nano: np.zeros(len(nano), dtype=np.float) - 9999.0 if args.data else nano.array("GenMET_pt"),
    "met_gen_phi": lambda nano: np.zeros(len(nano), dtype=np.float) - 9999.0 if args.data else nano.array("GenMET_phi"),
    "hasTau": lambda nano: nano.array("nTau") > 0,
    "firstgoodvertex": lambda nano: np.zeros(len(nano), dtype=np.int),
    "lumi": lambda nano: np.zeros(len(nano), dtype=np.float) + args.lumi,
    "evt_scale1fb": lambda nano: np.ones(len(nano), dtype=np.float),
    "evt_scale1fb": lambda nano: np.ones(len(nano), dtype=np.float),
    "xsec_br": lambda nano: np.ones(len(nano), dtype=np.float),
}

df_info = pd.DataFrame(index=sorted(all_branches))

for branch in all_branches:
    if not branch in converters:
        continue

    tgt = np.concatenate([converters[branch](nano)[nano_idx] for nano, nano_idx in zip(nanos, nano_indices)])
    ref = np.concatenate([baby.array(branch)[baby_idx] for baby_idx in baby_indices])

    df_info.loc[branch, "exact match [%]"] = (tgt == ref).sum() * 100.0 / len(tgt)
    df_info.loc[branch, "float match [%]"] = np.abs(((tgt - ref) / ref) < 1e-6).sum() * 100.0 / len(tgt)
    df_info.loc[branch, "bias [tgt - ref]"] = np.mean(tgt - ref)

print_df_repeated_header(df_info)
df_info.to_html("summary.html")
