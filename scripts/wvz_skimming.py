if __name__ == "__main__":

    import sys

    from geeksw.utils.data_loader_tools import make_data_loader, TreeWrapper, list_root_files_recursively

    import uproot
    import pandas as pd

    import libwwz

    import os

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

    skim = libwwz.skims.wvz_skim

    # python wvz_skimming.py /scratch/store/mc/RunIIFall17NanoAODv6/WWZJetsTo4L2Nu_4f_TuneCP5_13TeV_amcatnloFXFX_pythia8/NANOAODSIM/PU2017_12Apr2018_Nano25Oct2019_102X_mc2017_realistic_v7-v1 ../skims/2017_WWZ_for_jetmet_sync --overwrite --verbosity 2

    nano_files = list_root_files_recursively(args.input)

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
        "n_10_leptons",
        "n_25_leptons",
        "n_veto_leptons_noiso",
        "n_veto_leptons",
    ]

    columns = list(set(skim.deps + columns_to_save))

    data_loader = make_data_loader(columns, libwwz.producers.mc_producers, verbosity=args.verbosity)

    if not args.overwrite:
        for i_nano_file, nano_file in enumerate(nano_files):
            filename = os.path.basename(nano_file).split(".")[0]

            outfile = os.path.join(args.output, "parquet", str(i_nano_file) + "_" + filename + ".parquet")

            if os.path.isfile(outfile):
                print("I'm not doing the skim because " + outfile + " would be overwritten!")
                sys.exit()

    for i_nano_file, nano_file in enumerate(nano_files):
        filename = os.path.basename(nano_file).split(".")[0]
        print("skimming", filename)
        nano = TreeWrapper(uproot.open(nano_file)["Events"], n_max_events=None)

        data = skim(data_loader(nano))

        df = pd.DataFrame(data={c: data[c] for c in columns_to_save})

        outfile = os.path.join(args.output, "parquet", str(i_nano_file) + "_" + filename + ".parquet")

        df.to_parquet(outfile, compression="gzip", index=False)
