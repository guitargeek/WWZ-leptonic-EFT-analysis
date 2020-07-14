import libwwz

from geeksw.utils.core import concatenate
from geeksw.utils.data_loader_tools import make_data_loader, TreeWrapper, list_root_files_recursively

import uproot

import numpy as np
import matplotlib.pyplot as plt

baby_file = "/home/llr/cms/rembser/scratch/baby-ntuples/WVZMVA2017_v0.1.21/wwz_4l2v_amcatnlo_1.root"
nano_dir = "/scratch/store/mc/RunIIFall17NanoAODv6/WWZJetsTo4L2Nu_4f_TuneCP5_13TeV_amcatnloFXFX_pythia8/NANOAODSIM/PU2017_12Apr2018_Nano25Oct2019_102X_mc2017_realistic_v7-v1"

nano_files = list_root_files_recursively(nano_dir)

baby = uproot.open(baby_file)["t"]
baby_event = baby.array("evt")

print(baby.array("evt_scale1fb")[0])

all_branches = list(set([br.decode("utf-8") for br in baby.keys()]))

# nanos = [TreeWrapper(uproot.open(nano)["Events"], n_max_events=100) for nano in nano_files]


is_data = False
lumi = 0.0

columns = [
    "VetoNoIsoElectron_pt",
    "VetoNoIsoElectron_eta",
    "VetoNoIsoElectron_phi",
    "VetoNoIsoElectron_mass",
    "VetoNoIsoElectron_pfRelIso03_all",
    "VetoNoIsoElectron_pfRelIso03_all_wLep",
    "VetoNoIsoMuon_pt",
    "VetoNoIsoMuon_eta",
    "VetoNoIsoMuon_phi",
    "VetoNoIsoMuon_mass",
    "VetoNoIsoMuon_pfRelIso03_all",
    "VetoNoIsoMuon_pfRelIso03_all_wLep",
    *libwwz.output.columns,
]

data_loader = make_data_loader(columns, libwwz.producers.mc_producers, verbosity=0)

datas = []

skim = libwwz.skims.wvz_skim

for i_nano_file, nano_file in enumerate(nano_files):
    print(nano_file)
    nano = TreeWrapper(uproot.open(nano_file)["Events"], n_max_events=None)

    data = skim(data_loader(nano))

    datas.append(data)

data = dict()
for column in datas[-1].keys():
    data[column] = concatenate([d[column] for d in datas])

nano_event = data["evt"]

n_overlap = np.sum(np.in1d(nano_event, baby_event))
n_nano = len(nano_event)
n_baby = len(baby_event)

print()
print("Overlapping events:", n_overlap)
print("Events only in nano:", n_nano - n_overlap)
print("Events only in baby:", n_baby - n_overlap)
print()

_, nano_idx, baby_idx = np.intersect1d(nano_event, baby_event, return_indices=True)


def kinematics_comparison_plot(
    baby_variable="lep_pt", nano_variable="pt", bins=np.linspace(0, 200, 200), particle="Electron"
):

    particle = "VetoNoIso" + particle

    assert particle in ["VetoNoIsoElectron", "VetoNoIsoMuon"]

    lep_id = 11 if "Electron" in particle else 13

    electron_mask = np.abs(baby.array("lep_id")) == lep_id

    baby_values = baby.array(baby_variable)[electron_mask]
    baby_events = baby.array("evt")
    idx = baby_values.counts.argmax()
    label = particle + " " + baby_variable

    nano_values = data[particle + "_" + nano_variable]
    print(label)
    print("=" * len(label))
    print(f"Values in event {baby_events[idx]} (the event with the most objects)")
    print("baby: ", baby_values[idx])
    print("nano: ", nano_values[np.argmax(data["evt"] == baby_events[idx])])

    plt.hist(baby_values.flatten(), bins, histtype="step", label="BABY")
    plt.hist(nano_values.flatten(), bins, histtype="step", label="NANO")

    plt.legend(loc="upper right")
    # plt.gca().set_yscale("log", nonposy='clip')
    plt.xlabel(label)
    plt.ylabel("Events")
    plt.savefig(particle + "_" + baby_variable + ".png", dpi=300)
    # plt.show()
    plt.close()


pt_bins = np.linspace(0, 200, 200)
eta_bins = np.linspace(-3, 3, 200)

kinematics_comparison_plot("lep_pt", "pt", pt_bins, "Electron")
kinematics_comparison_plot("lep_eta", "eta", eta_bins, "Electron")
kinematics_comparison_plot("lep_pt", "pt", pt_bins, "Muon")
kinematics_comparison_plot("lep_eta", "eta", eta_bins, "Muon")

for particle in ["VetoNoIsoElectron", "VetoNoIsoMuon"]:

    lep_id = 11 if "Electron" in particle else 13

    electron_mask = np.abs(baby.array("lep_id")) == lep_id
    bins = np.linspace(0, 2, 200)

    baby_pt = baby.array("lep_pt")[electron_mask].flatten()

    for var in ["lep_relIso03EAv4", "lep_relIso03EAv4wLep"]:
        plt.hist(baby.array(var)[electron_mask].flatten(), bins, histtype="step", label="BABY - " + var)  # * baby_pt,

    for var in [particle + "_pfRelIso03_all", particle + "_pfRelIso03_all_wLep"]:
        plt.hist(data[var].flatten(), bins, histtype="step", label="NANO - lep_" + var[10:])

    plt.legend(loc="upper right")
    plt.gca().set_yscale("log", nonposy="clip")
    plt.title(particle)
    plt.xlabel("Relative Isolation")
    plt.ylabel("Events")
    plt.savefig(particle + "_isolation.png", dpi=300)
    plt.close()
    # plt.show()
