import numpy as np


def sortargs(pt):

    import awkward

    offsets = np.concatenate([[0], np.cumsum(pt.counts)[:-1]])

    assert (pt.counts == 4).all()

    masked_array = awkward.MaskedArray(mask=np.zeros(len(pt.flatten()), dtype=np.bool), content=pt.flatten())

    pt_masked = awkward.JaggedArray.fromcounts(pt.counts, masked_array)

    indices = []

    for i in range(4):
        indices.append(pt_masked.argmax())

        pt_masked.flatten().mask[indices[-1].flatten() + offsets] = True

    return indices[0].concatenate(indices[1:], axis=1)


def four_veto_lepton_df(data):
    inds = sortargs(data["VetoLepton_pt"])

    df = pd.DataFrame()

    pt_sorted = data["VetoLepton_pt"][inds]

    variables = ["pt", "eta", "phi", "mass", "pdgId", "ip3d", "sip3d", "pfRelIso03_all_wLep", "mediumId"]

    sorted_values = dict()

    sorted_p4 = data["VetoLepton_p4"][inds]

    for variable in variables:
        sorted_values[variable] = data["VetoLepton_" + variable][inds]

    for i in range(4):
        for variable in variables:
            df[f"VetoLepton_{variable}_{i}"] = sorted_values[variable][:, i]

    combinations = sorted_p4.cross(sorted_p4, nested=True)

    pair_masses = (combinations.i0 + combinations.i1).mass
    pair_masses.content.content = np.nan_to_num(pair_masses.content.content)

    for i in range(4):
        for j in range(4):
            if j <= i:
                continue
            df[f"VetoLeptonPair_mass_{i}{j}"] = pair_masses[:, i, j]

    return df


if __name__ == "__main__":

    import sys

    from geeksw.utils.data_loader_tools import make_data_loader, TreeWrapper, list_root_files_recursively

    import uproot
    import pandas as pd

    import libwwz

    import os
    import json

    import argparse

    parser = argparse.ArgumentParser(description="Configurable skimming of NanoAOD.")
    parser.add_argument("input", type=str, help="the path to the NanoAOD dataset")
    parser.add_argument("output", type=str, help="the path where the skims will be stored")
    parser.add_argument("--data", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbosity", type=int, default=0)
    parser.add_argument("--year", type=int, default=None)

    args = parser.parse_args()

    libwwz.config.year = args.year

    os.makedirs(os.path.join(args.output, "parquet"), exist_ok=True)

    skim = libwwz.skims.four_lepton_skim

    # python wvz_skimming.py /scratch/store/mc/RunIIFall17NanoAODv7/WWZJetsTo4L2Nu_4f_TuneCP5_13TeV_amcatnloFXFX_pythia8/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1 ../skims/2017_WWZ_four_lepton_skim --overwrite --verbosity 2 --year 2017

    nano_files = list_root_files_recursively(args.input)

    variables_for_df = ["pt", "eta", "phi", "mass", "pdgId", "ip3d", "sip3d", "pfRelIso03_all_wLep", "p4", "mediumId"]
    required_cols_for_df = ["VetoLepton_" + v for v in variables_for_df]

    columns_to_save = [
        "evt",
        "evt_passgoodrunlist",
        "evt_scale1fb",
        "firstgoodvertex",
        "hasTau",
        "isData",
        "lumi",
        "met_gen_phi",
        "met_gen_pt",
        "nTrueInt",
        "nvtx",
        "passesMETfiltersRun2",
        "run",
        "xsec_br",
        "nb",
        "genWeight",
        "MET_pt",
        "MET_phi",
    ]

    columns = list(set(skim.deps + columns_to_save)) + required_cols_for_df

    data_loader = make_data_loader(columns, libwwz.producers.mc_producers, verbosity=args.verbosity)

    if not args.overwrite:
        for i_nano_file, nano_file in enumerate(nano_files):
            filename = os.path.basename(nano_file).split(".")[0]

            outfile = os.path.join(args.output, "parquet", str(i_nano_file) + "_" + filename + ".parquet")

            if os.path.isfile(outfile):
                print("I'm not doing the skim because " + outfile + " would be overwritten!")
                sys.exit()

    metainfo = {"genWeightSum": 0.0, "genWeightSum2": 0.0}

    for i_nano_file, nano_file in enumerate(nano_files):
        filename = os.path.basename(nano_file).split(".")[0]
        print("skimming", filename)
        nano = TreeWrapper(uproot.open(nano_file)["Events"], n_max_events=None)

        data_full = data_loader(nano)

        metainfo["genWeightSum"] = metainfo["genWeightSum"] + np.sum(data_full["genWeight"])
        metainfo["genWeightSum2"] = metainfo["genWeightSum2"] + np.sum(data_full["genWeight"] ** 2)

        data = skim(data_full)

        if len(data["genWeight"]) == 0:
            continue

        df_leptons = four_veto_lepton_df(data)

        df_other = pd.DataFrame(data={c: data[c] for c in columns_to_save})

        df = pd.concat([df_other, df_leptons], axis=1)

        outfile = os.path.join(args.output, "parquet", str(i_nano_file) + "_" + filename + ".parquet")

        df.to_parquet(outfile, compression="gzip", index=False)

    with open(os.path.join(args.output, "metainfo.json"), "w") as outfile:
        outfile.write(json.dumps(metainfo, indent=4, sort_keys=True))
