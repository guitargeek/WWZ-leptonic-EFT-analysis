import numpy as np


def four_lepton_skim(data):

    skim_mask = np.logical_and.reduce(
        [
            data["n_10_leptons"] >= 4,
            data["n_25_leptons"] >= 2,
            data["n_veto_leptons_noiso"] >= 4,
            data["n_veto_leptons"] >= 2,
        ]
    )

    data_skimmed = dict()
    for column, array in data.items():

        data_skimmed[column] = array[skim_mask]

    return data_skimmed
