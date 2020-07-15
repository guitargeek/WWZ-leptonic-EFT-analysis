if __name__ == "__main__":

    import os
    import argparse

    parser = argparse.ArgumentParser(description="Copy datasets to scratch.")
    parser.add_argument("datasets", type=str, help="textfile with all NanoAOD datasets to copy")
    # parser.add_argument("output", type=str, help="the path where the skims will be stored")
    # parser.add_argument("--data", action="store_true")
    # parser.add_argument("--overwrite", action="store_true")
    # parser.add_argument("--verbosity", type=int, default=0)

    args = parser.parse_args()

    # server = "polgrid4.in2p3.fr"
    # server = "cms-gridftp.rcac.purdue.edu"
    server = "cms-xrd-global.cern.ch"
    # server = "cmsxrootd.fnal.gov"
    # server = "gridsrm.ts.infn.it"
    # server = "hephyse.oeaw.ac.at"
    # server = "cmssrm.hep.wisc.edu"
    # server = "cmsio.rc.ufl.edu"

    with open(args.datasets, "r") as f:
        datasets = [l.strip() for l in f.readlines() if not l.strip().startswith("#")]

    for ds in datasets:

        file_list = os.popen('dasgoclient -query="file dataset={0}"'.format(ds)).read()

        if not ".root" in file_list:
            print(ds + " has no files! Maybe it's not a valid dataset?")
            continue

        file_list = [f.strip() for f in file_list.split("\n") if ".root" in f]
        print(ds + f" has {len(file_list)} files")

        for fname in file_list:
            scratch_fname = "/scratch" + fname
            print(scratch_fname)
            if os.path.isfile(scratch_fname):
                print("Skipping already synchronized file: " + fname)
                continue
            print("Copying missing file: " + fname)
            copy_command = "xrdcp root://" + server + "/" + fname + " " + scratch_fname
            print(copy_command)
            os.system(copy_command)
