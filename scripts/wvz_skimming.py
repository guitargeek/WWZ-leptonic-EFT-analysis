if __name__ == "__main__":

    import argparse
    import sys

    from geeksw.utils.data_loader_tools import make_data_loader, TreeWrapper, list_root_files_recursively

    import uproot
    import pandas as pd

    import libwwz

    import os

    skim = libwwz.skims.four_lepton_skim

    parser = argparse.ArgumentParser(description="Configurable skimming of NanoAOD.")
    parser.add_argument("input", type=str, help="the path to the NanoAOD dataset")
    parser.add_argument("output", type=str, help="the path where the skims will be stored")
    parser.add_argument("--data", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbosity", type=int, default=0)

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # /scratch/store/mc/RunIIFall17NanoAODv6/WWZJetsTo4L2Nu_4f_TuneCP5_13TeV_amcatnloFXFX_pythia8/NANOAODSIM/PU2017_12Apr2018_Nano25Oct2019_102X_mc2017_realistic_v7-v1

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
    ]

    columns = list(set(libwwz.skims.four_lepton_skim_required_columns + columns_to_save))

    data_loader = make_data_loader(columns, libwwz.producers.mc_producers, verbosity=args.verbosity)

    if not args.overwrite:
        for i_nano_file, nano_file in enumerate(nano_files):
            filename = os.path.basename(nano_file).split(".")[0]

            outfile = os.path.join(args.output, filename + ".parquet")

            if os.path.isfile(outfile):
                print("I'm not doing the skim because " + outfile + " would be overwritten!")
                sys.exit()

    for i_nano_file, nano_file in enumerate(nano_files):
        filename = os.path.basename(nano_file).split(".")[0]
        print("skimming", filename)
        nano = TreeWrapper(uproot.open(nano_file)["Events"], n_max_events=None)

        data = skim(data_loader(nano))

        df = pd.DataFrame(data={c: data[c] for c in columns_to_save})

        outfile = os.path.join(args.output, filename + ".parquet")

        df.to_parquet(outfile, compression=None, index=False)
