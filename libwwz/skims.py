import numpy as np


def _get_wvz_skim_mask(data):

    return np.logical_and.reduce(
        [
            data["n_10_leptons"] >= 4,
            data["n_25_leptons"] >= 2,
            data["n_veto_leptons_noiso"] >= 4,
            data["n_veto_leptons"] >= 2,
        ]
    )


class Skim(object):
    def __init__(self, mask_function, dependencies):
        self._mask_function = mask_function
        self.deps = dependencies

    def __call__(self, data, return_mask=False):

        skim_mask = self._mask_function(data)

        data_skimmed = dict()
        for column, array in data.items():

            data_skimmed[column] = array[skim_mask]

        if return_mask:
            return data_skimmed, skim_mask

        return data_skimmed


wvz_skim = Skim(_get_wvz_skim_mask, ["n_10_leptons", "n_25_leptons", "n_veto_leptons_noiso", "n_veto_leptons"])

four_lepton_skim = Skim(lambda data: data["n_veto_leptons"] == 4, ["n_veto_leptons"])
