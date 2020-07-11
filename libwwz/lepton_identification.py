import numpy as np


def passes_very_loose_muon_id(muon_table):
    """Takes a Muon table and returns a mask corrsponding to the very loose muon ID.
    """

    selection_masks = [
        muon_table["Muon_looseId"] == 1,
        muon_table["Muon_pt"] > 10.0,
        np.abs(muon_table["Muon_eta"]) < 2.4,
        np.abs(muon_table["Muon_dz"]) < 0.1,
        np.abs(muon_table["Muon_dxy"]) < 0.05,
    ]

    mask = True
    for s in selection_masks:
        mask = np.logical_and(mask, s)

    return mask


def passes_loose_muon_pog_id(muon_table):
    # https://github.com/cmstas/CORE/blob/master/MuonSelections.cc

    # equivalent to "Muon_looseId"
    mask = np.logical_and(
        muon_table["Muon_isPFcand"], np.logical_or(muon_table["Muon_isGlobal"], muon_table["Muon_isTracker"])
    )

    return mask


def passes_very_loose_electron_id(electron_table):

    selection_masks = [
        electron_table["Electron_pt"] > 10.0,
        np.abs(electron_table["Electron_eta"]) < 2.5,
        np.abs(electron_table["Electron_dz"]) < 0.1,
        np.abs(electron_table["Electron_dxy"]) < 0.05,
        electron_table["Electron_mvaFall17V2noIso_WPL"] == 1,
    ]

    mask = True
    for s in selection_masks:
        mask = np.logical_and(mask, s)

    return mask
