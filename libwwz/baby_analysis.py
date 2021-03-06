from libwwz.pileup_reweighting import get_pileup_weights


def event_weight(wvz, sample_name, year, tag, pileup_reweighting=False):

    fix_xsec = 1.0

    if sample_name == "ggh_hzz4l_powheg_1":
        fix_xsec = 1.1287633316  # Difference between scale1fb and HXSWG twiki
    if sample_name == "ggh_hzz4l_powheg_1" and year == 2017:
        fix_xsec = 1.1287633316 * 1.236e-05 / 5.617e-05  # Difference between scale1fb and HXSWG twiki
    if sample_name == "zz_4l_powheg_1":
        fix_xsec = 1.1  # Missing K-factor (scale1fb set to 1.256 which is without kfactor)
    if sample_name == "ttz_llvv_mll10":
        fix_xsec = 0.2728 / 0.2529  # TTZ AN2018-025 has 0.2728 while we used 0.2529
    if sample_name == "wwz_4l2v_amcatnlo_1" and "v0.1.15" in tag and year == 2018:
        fix_xsec = 3.528723e-7 / 3.1019e-7  #  error from wrong scale1fb

    if year == 2016:
        evt_weights = fix_xsec * wvz["evt_scale1fb"] * 35.9
    elif year == 2017:
        evt_weights = fix_xsec * wvz["evt_scale1fb"] * 41.3
    elif year == 2018:
        evt_weights = fix_xsec * wvz["evt_scale1fb"] * 59.74
    else:
        evt_weights = fix_xsec * wvz["evt_scale1fb"] * 137.0

    if pileup_reweighting:
        evt_weights *= get_pileup_weights(wvz["nTrueInt"], year)

    # isData column is still somehow bugged so we don't touch it
    # evt_weights[wvz["isData"]] = 1.0

    return evt_weights


def run_baby_analysis():
    import uproot
    import uproot_methods
    import numpy as np
    import pandas as pd

    from libwwz.array_utils import reduce_and, reduce_or
    from libwwz.array_utils import awkward_indices
    from libwwz.cut_utils import add_cut
    from libwwz.physics import find_z_pairs
    import libwwz.scale_factors as scale_factors

    import libwwz

    from geeksw.utils.data_loader_tools import TreeWrapper

    def cut_gen_filter(df, sample_name, year):
        if "wwz_amcatnlo" in sample_name and year != 2016:
            return df["nLightLep"] == 4
        return True

    def pass_vefrom_electron_id(wvz):
        return reduce_and(
            np.abs(wvz["lep_id"]) == 11,
            wvz["lep_isCutBasedIsoVetoPOG"] == 1,
            np.abs(wvz["lep_sip3d"]) < 4.0,
            wvz["lep_pt"] > 10.0,
            np.abs(wvz["lep_eta"]) < 2.5,
        )

    def pass_vefrom_muon_id(wwz):
        return reduce_and(
            np.abs(wwz["lep_id"]) == 13,
            wwz["lep_isMediumPOG"] == 1,
            np.abs(wwz["lep_sip3d"]) < 4.0,
            wwz["lep_pt"] > 10.0,
            wwz["lep_relIso04DB"] < 0.25,
            np.abs(wwz["lep_eta"]) < 2.4,
        )

    def pass_vefrom_lepton_id(wvz):
        return reduce_or(pass_vefrom_electron_id(wvz), pass_vefrom_muon_id(wvz))

    def pass_vefrom_lepton_mva_id(wvz):
        return wvz["lep_isVVVVeto"] > 0

    def pass_nominal_lepton_id(wvz):
        pdg = np.abs(wvz["lep_id"])
        pass_el = reduce_and(pdg == 11, wvz["lep_isCutBasedIsoMediumPOG"])
        pass_mu = reduce_and(pdg == 13, wvz["lep_relIso04DB"] < 0.15)
        return reduce_and(wvz["lep_pass_veto"], reduce_or(pass_el, pass_mu))

    def pass_z_lepton_mva_id(wvz):
        pdg = np.abs(wvz["lep_id"])
        pass_el = reduce_and(
            pdg == 11,
            # wvz["lep_isMVAwp90IsoPOG"],
            np.abs(wvz["lep_sip3d"]) < 4.0,
            wvz["lep_relIso03EAwLep"] < 0.2,
            wvz["lep_isMVAwpLooseNoIsoPOG"],
        )
        pass_mu = reduce_and(
            pdg == 13, wvz["lep_isMediumPOG"], np.abs(wvz["lep_relIso04DB"] < 0.25), np.abs(wvz["lep_sip3d"]) < 4.0
        )
        return reduce_and(wvz["lep_pass_veto"], reduce_or(pass_el, pass_mu))

    def pass_nominal_lepton_mva_id(wvz):
        pdg = np.abs(wvz["lep_id"])
        pass_el = reduce_and(
            pdg == 11,
            wvz["lep_isMVAwp90IsoPOG"],
            np.abs(wvz["lep_sip3d"]) < 4.0,
            wvz["lep_relIso03EAwLep"] < 0.2,
            wvz["lep_isMVAwpLooseNoIsoPOG"],
        )
        pass_mu = reduce_and(
            pdg == 13, wvz["lep_isMediumPOG"], np.abs(wvz["lep_relIso04DB"] < 0.15), np.abs(wvz["lep_sip3d"]) < 4.0
        )
        return reduce_and(wvz["lep_pass_veto"], reduce_or(pass_el, pass_mu))

    def is_4_lepton_event(wvz):
        m = reduce_and(wvz["lep_pass_veto"].sum() == 4, np.sign(wvz["lep_id"][wvz["lep_pass_veto"]]).sum() == 0)
        pdg = wvz["lep_id"][wvz["lep_pass_veto"]][m]
        m[m] = reduce_or(
            pdg[:, 0] + pdg[:, 1] == 0,
            pdg[:, 0] + pdg[:, 2] == 0,
            pdg[:, 0] + pdg[:, 3] == 0,
            pdg[:, 1] + pdg[:, 2] == 0,
            pdg[:, 1] + pdg[:, 3] == 0,
            pdg[:, 2] + pdg[:, 3] == 0,
        )
        return m

    def cut_hlt(wvz):
        presel = reduce_and(
            wvz["lep_pass_veto"].sum() >= 2,
            wvz["firstgoodvertex"] == 0,
            wvz["passesMETfiltersRun2"],
            np.logical_or(~wvz["isData"], wvz["pass_duplicate_mm_em_ee"]),
        )

        # Check if any of the combination of leptons pass the trigger thresholds
        # Ele 23 12
        # El23 Mu8
        # Mu23 El12
        # Mu 17 8
        # The thresholds are rounded up to 25, 15, or 10

        pt = wvz["lep_pt"][wvz["lep_pass_veto"]]
        pdg = np.abs(wvz["lep_id"][wvz["lep_pass_veto"]])

        ele_pt = pt[pdg == 11]
        mu_pt = pt[pdg == 13]
        triggered = reduce_or(
            reduce_and(wvz["HLT_DoubleEl"], (ele_pt > 25).sum(), (ele_pt > 15).sum() > 1),
            reduce_and(wvz["HLT_MuEG"], (mu_pt > 25).sum(), (ele_pt > 15).sum()),
            reduce_and(wvz["HLT_MuEG"], (ele_pt > 25).sum(), (mu_pt > 10).sum()),
            reduce_and(wvz["HLT_DoubleMu"], (mu_pt > 20).sum(), (mu_pt > 10).sum() > 1),
        )

        return np.logical_and(presel, triggered)

    year = 2017
    sample_name = "wwz_4l2v_amcatnlo_1"
    # year = 2016
    # sample_name = "wwz_amcatnlo_1"

    tag = "WVZMVA{0}_v0.1.21".format(year)
    # tag = "WVZMVA{0}_v0.1.15".format(year)
    # tag = "WVZ{0}_v0.1.12.7".format(year)

    root_file_name = "/scratch/rembser/babies/{0}/{1}.root".format(tag, sample_name)
    t = uproot.open(root_file_name)["t"]

    entrystop = 10000
    wvz = TreeWrapper(t, n_max_events=entrystop, extendable=True)

    wvz["lep_mass"] = 0.0

    wvz["lep_p4_"] = uproot_methods.TLorentzVectorArray.from_ptetaphim(
        wvz["lep_pt"], wvz["lep_eta"], wvz["lep_phi"], wvz["lep_mass"]
    )

    use_mva = True

    if use_mva:
        wvz["lep_pass_veto"] = pass_vefrom_lepton_mva_id(wvz)
        wvz["lep_pass_z_id"] = pass_z_lepton_mva_id(wvz)
        wvz["lep_pass_nominal"] = pass_nominal_lepton_mva_id(wvz)
    else:
        wvz["lep_pass_veto"] = pass_vefrom_lepton_id(wvz)
        wvz["lep_pass_z_id"] = wvz["lep_pass_veto"]
        wvz["lep_pass_nominal"] = pass_nominal_lepton_id(wvz)

    wvz["evt_weight"] = libwwz.baby_analysis.event_weight(wvz, sample_name, year, tag)
    wvz["lep_idx_"] = awkward_indices(wvz["lep_pt"])

    lep_z_id_is_z = find_z_pairs(wvz["lep_p4_"][wvz["lep_pass_z_id"]], wvz["lep_id"][wvz["lep_pass_z_id"]])

    wvz["lep_is_z"] = wvz["lep_pt"] < 0.0
    wvz["lep_is_z"][wvz["lep_idx_"][wvz["lep_pass_z_id"]][lep_z_id_is_z]] = True

    wvz["z_mass"] = wvz["lep_p4_"][wvz["lep_is_z"]].sum().mass

    wvz["lep_is_nom"] = np.logical_and(~wvz["lep_is_z"], wvz["lep_pass_nominal"])

    wvz["lep_et"] = wvz["lep_energy"] / np.cosh(wvz["lep_eta"])
    wvz["lep_mt"] = np.sqrt(2.0 * wvz["met_pt"] * wvz["lep_et"] * (1.0 - np.cos(wvz["lep_phi"] - wvz["met_phi"])))

    def cut_four_leptons_low_mll(wvz):
        mask = reduce_or(wvz["lep_is_z"], wvz["lep_is_nom"])

        res = mask.sum() == 4

        p4 = wvz["lep_p4_"][mask][res]
        ch = np.sign(wvz["lep_id"][mask][res])

        mll_cut = 12.0

        combinations = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

        def veto(i, j):
            return reduce_and(ch[:, i] == -ch[:, j], (p4[:, i] + p4[:, j]).mass < mll_cut)

        res[res] = ~reduce_or(*[veto(i, j) for i, j in combinations])

        return res

    def cut_four_lepton_pt(wvz):
        pt = wvz["lep_pt"]
        return reduce_and(pt[wvz["lep_is_z"]].max() > 25.0, pt[wvz["lep_is_nom"]].max() > 25.0)

    wvz["has_two_lep_nom"] = reduce_and(
        wvz["lep_is_z"].sum() == 2, wvz["lep_is_nom"].sum() == 2, wvz["lep_id"][wvz["lep_is_nom"]].prod() < 0.0
    )

    wvz["is_SFOS"] = wvz["lep_id"][wvz["lep_is_nom"]].prod() != -143
    wvz["is_on_z"] = np.logical_and(
        np.abs(wvz["lep_p4_"][wvz["lep_is_nom"]].sum().mass - 91.1876) < 10.0,
        wvz["lep_id"][wvz["lep_is_nom"]].sum() == 0,
    )

    wvz["is_high_mt"] = np.logical_and(
        (wvz["lep_mt"][wvz["lep_is_nom"]] > 40).sum() >= 1, (wvz["lep_mt"][wvz["lep_is_nom"]] > 20).sum() >= 2
    )

    # Start with cutflow
    tgt = pd.DataFrame(index=np.arange(len(wvz["evt"])))
    tgt = add_cut(tgt, "Root", True)
    tgt = add_cut(tgt, "EventWeight", True, previous_cut="Root", weight=wvz["evt_weight"])
    tgt = add_cut(tgt, "GenFilter", cut_gen_filter(wvz, sample_name, year), previous_cut="EventWeight")

    # List of common four lepton related selections
    tgt = add_cut(tgt, "Weight", True, previous_cut="GenFilter")

    def ele_z_cand_sf(wvz):
        m = reduce_and(wvz["lep_is_z"], np.abs(wvz["lep_id"]) == 11)
        return scale_factors.elec_reco_sf[year](wvz["lep_pt"][m], wvz["lep_eta"][m]).prod()

    def ele_nom_sf(wvz):
        m = reduce_and(wvz["lep_is_nom"], np.abs(wvz["lep_id"]) == 11)
        reco_sf = scale_factors.elec_reco_sf[year](wvz["lep_pt"][m], wvz["lep_eta"][m])
        id_sf = scale_factors.elec_mva_medium_sf[year](wvz["lep_pt"][m], wvz["lep_eta"][m])

        return (reco_sf * id_sf).prod()

    def muon_z_cand_sf(wvz, year=None):
        m = reduce_and(wvz["lep_is_z"], np.abs(wvz["lep_id"]) == 13)
        reco_sf = scale_factors.muon_reco_sf[year](wvz["lep_pt"][m], wvz["lep_eta"][m])

        pt_min = 20.1
        if year == 2018:
            pt_min = 15.1
        id_sf = scale_factors.muon_looseiso_sf[year](np.maximum(wvz["lep_pt"][m], pt_min), wvz["lep_eta"][m])

        return (reco_sf * id_sf).prod()

    def muon_nom_sf(wvz, year=None):
        m = reduce_and(wvz["lep_is_nom"], np.abs(wvz["lep_id"]) == 13)
        reco_sf = scale_factors.muon_reco_sf[year](wvz["lep_pt"][m], wvz["lep_eta"][m])

        pt_min = 20.1
        if year == 2018:
            pt_min = 15.1
        id_sf = scale_factors.muon_tightiso_sf[year](np.maximum(wvz["lep_pt"][m], pt_min), wvz["lep_eta"][m])

        return (reco_sf * id_sf).prod()

    tgt = add_cut(tgt, "FourLeptons", is_4_lepton_event(wvz), previous_cut="Weight")

    tgt = add_cut(tgt, "ElectronZCandSF", True, previous_cut="FourLeptons", weight=ele_z_cand_sf(wvz))
    tgt = add_cut(tgt, "MuonZCandSF", True, previous_cut="ElectronZCandSF", weight=muon_z_cand_sf(wvz, year=year))
    tgt = add_cut(tgt, "ElectronNomSF", True, previous_cut="MuonZCandSF", weight=ele_nom_sf(wvz))
    tgt = add_cut(tgt, "MuonNomSF", True, previous_cut="ElectronNomSF", weight=muon_nom_sf(wvz, year=year))

    tgt = add_cut(tgt, "CutHLT", cut_hlt(wvz), previous_cut="MuonNomSF")
    tgt = add_cut(tgt, "FindTwoOSNominalLeptons", wvz["has_two_lep_nom"], previous_cut="CutHLT")
    tgt = add_cut(tgt, "Cut4LepLowMll", cut_four_leptons_low_mll(wvz), previous_cut="FindTwoOSNominalLeptons")
    tgt = add_cut(tgt, "Cut4LepLeptonPt", cut_four_lepton_pt(wvz), previous_cut="Cut4LepLowMll")
    tgt = add_cut(tgt, "FindZCandLeptons", True, previous_cut="Cut4LepLeptonPt")
    tgt = add_cut(tgt, "Cut4LepBVeto", wvz["nb"] == 0, previous_cut="FindZCandLeptons", weight=wvz["weight_btagsf"])

    # emu channel
    tgt = add_cut(tgt, "ChannelEMu", ~wvz["is_SFOS"], previous_cut="Cut4LepBVeto")

    # OnZ channel
    tgt = add_cut(tgt, "ChannelOnZ", wvz["is_on_z"], previous_cut="Cut4LepBVeto")
    tgt = add_cut(tgt, "ChannelOnZNjet", wvz["nj"] >= 2, previous_cut="ChannelOnZ")

    # OffZ channel
    tgt = add_cut(tgt, "ChannelOffZ", np.logical_and(~wvz["is_on_z"], wvz["is_SFOS"]), previous_cut="Cut4LepBVeto")
    tgt = add_cut(tgt, "ChannelOffZHighMET", wvz["met_pt"] > 100.0, previous_cut="ChannelOffZ")
    tgt = add_cut(tgt, "ChannelOffZLowMET", wvz["met_pt"] <= 100.0, previous_cut="ChannelOffZ")
    tgt = add_cut(tgt, "ChannelOffZHighMT", wvz["is_high_mt"], previous_cut="ChannelOffZ")
    tgt = add_cut(tgt, "ChannelOffZLowMT", ~wvz["is_high_mt"], previous_cut="ChannelOffZ")

    return tgt


def get_yields_after_cuts(data_frame, cuts):
    series = data_frame[map(lambda s: s + "Weight", cuts)].sum()
    series.index = map(lambda s: s[:-6], series.index)
    return series
