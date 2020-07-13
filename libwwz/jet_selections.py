import numpy as np


def passes_tight_jet_id(data):
    cuts = [
        data["Jet_nConstituents"] > 1,
        data["Jet_neHEF"] < 0.9,
        data["Jet_neEmEF"] < 0.9,
        data["Jet_chHEF"] > 0.0,
        data["Jet_chHEF"] + data["Jet_chEmEF"] > 0.0,
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
        data["Jet_passes_tight_id"],
        data["Jet_pt"] > 30.0,
        np.abs(data["Jet_eta"]) < 2.4,
    ]

    out = cuts[0]

    for i in range(1, len(cuts)):
        out = np.logical_and(out, cuts[i])

    return out


def passes_vvv_b_jet_selection(data):

    cuts = [
        data["Jet_btagDeepB"] > 0.1522,
        data["Jet_passes_tight_id"],
        data["Jet_pt"] > 20,
        np.abs(data["Jet_eta"]) < 2.4,
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
    """
    idx_offset = np.concatenate([[0], np.cumsum(match_selection.counts)[:-1]])
    idx_offset = np.array(idx_offset, dtype=np.int32)
    match_idx_flat = (match_idx + idx_offset).flatten()

    return match_selection.flatten()[match_idx_flat] & (match_idx.flatten() >= 0)
