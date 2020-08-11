import uproot
import pandas as pd
import numpy as np
import os

import awkward
import uproot_methods

# from root_utils import RootHistogramWriter
import libwwz

tgt = libwwz.baby_analysis.run_baby_analysis()

from_validate = [
    "Root",
    "EventWeight",
    "GenFilter",
    "Weight",
    "FourLeptons",
    # "ElectronZCandSF",
    # "MuonZCandSF",
    # "ElectronNomSF",
    # "MuonNomSF",
    "CutHLT",
    "FindTwoOSNominalLeptons",
    "Cut4LepLowMll",
    "Cut4LepLeptonPt",
    "FindZCandLeptons",
    "Cut4LepBVeto",
    "ChannelEMu",
    "ChannelOnZ",
    "ChannelOnZNjet",
    "ChannelOffZ",
    "ChannelOffZ",
    "ChannelOffZLowMET",
    "ChannelOffZHighMET",
    "ChannelOffZLowMT",
    "ChannelOffZHighMT",
]

df_compare = pd.DataFrame()

compare_with_ref = False

if compare_with_ref:
    ref_file = uproot.open("../tmp/WVZLooper/outputs/" + tag + "/test/CutResults_MC_" + sample_name + "_results.root")
    ref = ref_file["cutResultTree"].pandas.df(entrystop=entrystop)

    d = 0 + tgt[from_validate[-1]] - ref[from_validate[-1]]
    mask = tgt[from_validate[-1]] ^ ref[from_validate[-1]]
    print(d.value_counts())

    df_compare["agreement [%]"] = (tgt[from_validate] == ref[from_validate]).sum() * 1.0 / len(tgt) * 100
    df_compare["kept only by jonas [%]"] = (tgt[from_validate] > ref[from_validate]).sum() * 1.0 / len(tgt) * 100
    df_compare["kept only by philip [%]"] = (tgt[from_validate] < ref[from_validate]).sum() * 1.0 / len(tgt) * 100

    series = ref[map(lambda s: s + "Weight", from_validate)].sum()
    series.index = map(lambda s: s[:-6], series.index)
    df_compare["yield philip"] = series

df_compare["yield jonas"] = libwwz.baby_analysis.get_yields_after_cuts(tgt, from_validate)

series = tgt[from_validate].sum()
df_compare["n jonas"] = series

print("")
print("Syncronization level:")
print(df_compare)
print("")

# hist_writer = RootHistogramWriter("out.root")
# for cut in from_validate:
# hist_writer.fill(cut + "__Njet", 6, 0, 6, wvz["met_pt"][tgt[cut]], weights=tgt[cut + "Weight"])
# hist_writer.write()


# tgt = uproot.open("out.root")
# ref = uproot.open("../tmp/WVZLooper/outputs/"+tag+"/test/MC_wwz_4l2v_amcatnlo_1_results.root")

# np.testing.assert_almost_equal(tgt["Weight__Njet"].alledges, ref["Weight__Njet"].alledges)
# np.testing.assert_almost_equal(tgt["Weight__Njet"].allvalues, ref["Weight__Njet"].allvalues)
# np.testing.assert_almost_equal(tgt["Weight__Njet"].allvariances, ref["Weight__Njet"].allvariances)
