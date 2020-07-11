from libwwz import lepton_identification

import numpy as np


def passes_muon_veto_id_noiso(data):
    muon_mask_1 = lepton_identification.passes_loose_muon_pog_id(data)
    muon_mask_2 = lepton_identification.passes_very_loose_muon_id(data)

    return np.logical_and(muon_mask_1, muon_mask_2)


def make_lepton_counter(pt_threshold=0.0):
    def producer(data):
        ele_pt = data["Electron_pt"]
        muon_pt = data["Muon_pt"]
        ele_abseta = np.abs(data["Electron_eta"])
        muon_abseta = np.abs(data["Muon_eta"])

        return (
            np.logical_and(ele_pt > pt_threshold, ele_abseta < 2.5).sum()
            + np.logical_and(muon_pt > pt_threshold, muon_abseta < 2.4).sum()
        )

    return producer


def make_lorentz_vector_producer(collection_name):

    import uproot_methods

    def produce(data):

        return uproot_methods.TLorentzVectorArray.from_ptetaphim(
            data[collection_name + "_pt"],
            data[collection_name + "_eta"],
            data[collection_name + "_phi"],
            data[collection_name + "_mass"],
        )

    return produce


def rel_iso_component(leptons, pfcands):
    pairing = leptons.cross(pfcands, nested=True)
    delta_r = pairing.i0.delta_r(pairing.i1)
    in_cone = np.logical_and(delta_r < 0.3, delta_r > 0.0005)

    return (pairing.i1[in_cone].pt / pairing.i0[in_cone].pt).sum()


def electron_isolation_with_pf_leptons(data):
    electrons = data["UncorrElectron_p4"]
    muons = data["Muon_p4"]

    pf_electrons = electrons[data["Electron_isPFcand"]]
    pf_muons = muons[data["Muon_isPFcand"]]

    return (
        data["Electron_pfRelIso03_all"]
        + rel_iso_component(electrons, pf_electrons)
        + rel_iso_component(electrons, pf_muons)
    )


def muon_isolation_with_pf_leptons(data):
    electrons = data["UncorrElectron_p4"]
    muons = data["Muon_p4"]

    pf_electrons = electrons[data["Electron_isPFcand"]]
    pf_muons = muons[data["Muon_isPFcand"]]

    return data["Muon_pfRelIso03_all"] + rel_iso_component(muons, pf_electrons) + rel_iso_component(muons, pf_muons)


common_producers = {
    "lumi": lambda nano: nano["luminosityBlock"],
    "evt": lambda nano: nano["event"],
    "evt_passgoodrunlist": lambda nano: np.ones(len(nano["event"]), dtype=np.bool),
    "evt_firstgoodvertex": lambda nano: np.ones(len(nano["event"]), dtype=np.int),
    "nvtx": lambda nano: nano["PV_npvsGood"],
    "passesMETfiltersRun2": lambda nano: nano["Flag_METFilters"],
    "hasTau": lambda nano: nano["nTau"] > 0,
    "Electron_pt_orig": lambda nano: nano["Electron_pt"] / nano["Electron_eCorr"],
    "firstgoodvertex": lambda nano: np.zeros(len(nano["event"]), dtype=np.int),
    "evt_scale1fb": lambda nano: np.ones(len(nano["event"]), dtype=np.float),
    "evt_scale1fb": lambda nano: np.ones(len(nano["event"]), dtype=np.float),
    "xsec_br": lambda nano: np.ones(len(nano["event"]), dtype=np.float),
    # Uncorrected electron
    "UncorrElectron_pt": lambda nano: nano["Electron_pt"] / nano["Electron_eCorr"],
    "UncorrElectron_eta": lambda nano: nano["Electron_eta"],
    "UncorrElectron_phi": lambda nano: nano["Electron_phi"],
    "UncorrElectron_mass": lambda nano: nano["Electron_mass"],
    # Lepton four vector
    "Elecron_p4": make_lorentz_vector_producer("Electron"),
    "UncorrElectron_p4": make_lorentz_vector_producer("UncorrElectron"),
    "Muon_p4": make_lorentz_vector_producer("Muon"),
    # Custom Isolation
    "Electron_relIso03EAv4wLep": electron_isolation_with_pf_leptons,
    "Muon_relIso03EAv4wLep": muon_isolation_with_pf_leptons,
    "Electron_veto_mask_noiso": lepton_identification.passes_very_loose_electron_id,
    "Electron_veto_mask": lambda d: np.logical_and(d["Electron_veto_mask_noiso"], d["Electron_relIso03EAv4wLep"] < 0.4),
    "Muon_veto_mask_noiso": passes_muon_veto_id_noiso,
    "Muon_veto_mask": lambda d: np.logical_and(d["Muon_veto_mask_noiso"], d["Muon_relIso03EAv4wLep"] < 0.4),
    # counting stuff
    "n_veto_leptons_noiso": lambda d: d["Electron_veto_mask_noiso"].sum() + d["Muon_veto_mask_noiso"].sum(),
    "n_veto_leptons": lambda d: d["Electron_veto_mask"].sum() + d["Muon_veto_mask"].sum(),
    "n_10_leptons": make_lepton_counter(pt_threshold=10.0),
    "n_25_leptons": make_lepton_counter(pt_threshold=25.0),
}

data_producers = {
    "isData": lambda nano: np.zeros(len(nano["event"]), dtype=np.bool) | True,
    "nTrueInt": lambda nano: np.zeros(len(nano["event"]), dtype=np.int) - 999,
    "met_gen_pt": lambda nano: np.zeros(len(nano["event"]), dtype=np.float) - 9999.0,
    "met_gen_phi": lambda nano: np.zeros(len(nano["event"]), dtype=np.float) - 9999.0,
    **common_producers,
}
mc_producers = {
    "isData": lambda nano: np.zeros(len(nano["event"]), dtype=np.bool) | False,
    "nTrueInt": lambda nano: nano["Pileup_nTrueInt"],
    "met_gen_pt": lambda nano: nano["GenMET_pt"],
    "met_gen_phi": lambda nano: nano["GenMET_phi"],
    **common_producers,
}
