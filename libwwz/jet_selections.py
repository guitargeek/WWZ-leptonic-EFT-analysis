import numpy as np

from libwwz.config import cfg


def passes_jet_id(data):
    cuts = [
        data["Jet_nConstituents"] >= cfg["jet_id_min_nConstituents"],
        data["Jet_neHEF"] < cfg["jet_id_max_neHEF"],
        data["Jet_neEmEF"] < cfg["jet_id_max_neEmEF"],
        data["Jet_chHEF"] > cfg["jet_id_min_chHEF"],
        data["Jet_chEmEF"] < cfg["jet_id_max_chEmEF"],
        data["Jet_chHEF"] + data["Jet_chEmEF"] > cfg["jet_id_min_ch_nConstituents"],
        # last cut should correspond to charged multiplicity > 0
    ]

    out = cuts[0]

    for i in range(1, len(cuts)):
        out = np.logical_and(out, cuts[i])

    return out


def jet_matches_veto_lepton(data):
    from geeksw.physics import match

    return match(data["Jet_p4"], data["VetoElectron_p4"]) | match(data["Jet_p4"], data["VetoMuon_p4"])


def passes_vvv_jet_id(data):

    cuts = [
        ~data["Jet_matches_veto_lepton"],
        data["Jet_passes_id"],
        data["Jet_pt"] > cfg["jet_min_pt"],
        np.abs(data["Jet_eta"]) < cfg["jet_max_eta"],
    ]

    out = cuts[0]

    for i in range(1, len(cuts)):
        out = np.logical_and(out, cuts[i])

    return out


def passes_vvv_b_jet_selection(data):

    cuts = [
        data["Jet_btagDeepB"] > cfg["jet_btagDeepB_cut"],
        data["Jet_passes_id"],
        data["Jet_pt"] > cfg["b_jet_min_pt"],
        np.abs(data["Jet_eta"]) < cfg["jet_max_eta"],
        ~data["Jet_matches_veto_lepton"],
    ]

    out = cuts[0]

    for i in range(1, len(cuts)):
        out = np.logical_and(out, cuts[i])

    return out


def _match_passing_selection(match_idx, match_selection):
    """

    Example:

    >>> match_passing_selection(data["Jet_electronIdx1"], data["Electron_veto_mask"]).sum()

    Note:

    * this is buggy due to the buggy mathing in NanoAOD
    """
    idx_offset = np.concatenate([[0], np.cumsum(match_selection.counts)[:-1]])
    idx_offset = np.array(idx_offset, dtype=np.int32)
    match_idx_flat = (match_idx + idx_offset).flatten()

    return match_selection.flatten()[match_idx_flat] & (match_idx.flatten() >= 0)
