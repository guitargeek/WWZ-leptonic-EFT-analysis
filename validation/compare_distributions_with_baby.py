import libwwz
from libwwz.array_utils import *

from geeksw.utils.core import concatenate
from geeksw.utils.data_loader_tools import make_data_loader, TreeWrapper, list_root_files_recursively

import uproot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

year = 2018
libwwz.config.year = year

if libwwz.config.year == 2016:
    # baby_file = "/home/llr/cms/rembser/scratch/baby-ntuples/WVZMVA2016_v0.1.21/wwz_amcatnlo_1.root"
    baby_file = "/home/llr/cms/rembser/scratch/baby-ntuples/WVZMVA2016_v0.1.21/ttz_llvv_mll10_amcatnlo_1.root"
    # nano_dir = "/scratch/store/mc/RunIISummer16NanoAODv7/WWZ_TuneCUETP8M1_13TeV-amcatnlo-pythia8/NANOAODSIM/PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1"
    nano_dir = "/scratch/store/mc/RunIISummer16NanoAODv7/TTZToLLNuNu_M-10_TuneCUETP8M1_13TeV-amcatnlo-pythia8/NANOAODSIM/PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8_ext2-v1"

if libwwz.config.year == 2017:
    # baby_file = "/home/llr/cms/rembser/scratch/baby-ntuples/WVZMVA2017_v0.1.21/wwz_4l2v_amcatnlo_1.root"
    baby_file = "/home/llr/cms/rembser/scratch/baby-ntuples/WVZMVA2017_v0.1.21/ttz_llvv_mll10_amcatnlo_1.root"
    # nano_dir = "/scratch/store/mc/RunIIFall17NanoAODv7/WWZJetsTo4L2Nu_4f_TuneCP5_13TeV_amcatnloFXFX_pythia8/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1"
    nano_dir = "/scratch/store/mc/RunIIFall17NanoAODv7/TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1"

if libwwz.config.year == 2018:
    # baby_file = "/home/llr/cms/rembser/scratch/baby-ntuples/WVZMVA2018_v0.1.21/wwz_4l2v_amcatnlo_1.root"
    baby_file = "/home/llr/cms/rembser/scratch/baby-ntuples/WVZMVA2018_v0.1.21/ttz_llvv_mll10_amcatnlo_1.root"
    # nano_dir = "/scratch/store/mc/RunIIAutumn18NanoAODv7/WWZJetsTo4L2Nu_4f_TuneCP5_13TeV_amcatnloFXFX_pythia8/NANOAODSIM/Nano02Apr2020_102X_upgrade2018_realistic_v21-v1"
    nano_dir = "/scratch/store/mc/RunIIAutumn18NanoAODv7/TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/Nano02Apr2020_102X_upgrade2018_realistic_v21_ext1-v1"

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
    "nb",
    "n_veto_leptons",
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

nano_idx, baby_idx = libwwz.validation.nano_baby_overlap(nano_event, baby_event)

def check_n_veto_leptons(baby, nano):

    # tree = TreeWrapper(baby)
    s1 = pd.Series(baby.array("lep_isVVVVeto")[baby_idx].sum())# + pass_vefrom_muon_id(tree)[baby_idx].sum())
    print(s1.value_counts())
    s2 = pd.Series(nano["n_veto_leptons"][nano_idx])
    print(s2.value_counts())

check_n_veto_leptons(baby, data)


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
    plt.savefig(str(year) + "_" + particle + "_" + baby_variable + ".png", dpi=300)
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
    plt.savefig(str(year) + "_" + particle + "_isolation.png", dpi=300)
    plt.close()
    # plt.show()

# Jet variables
bins = np.linspace(0, 10, 11)
plt.hist(baby.array("nb")[baby_idx], bins=bins, histtype="step", label="BABY")
plt.hist(data["nb"][nano_idx], bins=bins, histtype="step", label="NANO")
plt.legend(loc="upper right")
plt.xlabel("Number of b-jets")
plt.ylabel("Events")
plt.savefig(str(year) + "_" + "nb.png", dpi=300)
plt.close()
