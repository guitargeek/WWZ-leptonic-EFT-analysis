import ROOT
import root_numpy
import numpy as np


def as_float_array(array):
    if not (isinstance(array, np.ndarray) and array.dtype == np.float):
        array = np.array(array, dtype=np.float)
    return array


class RootHistogramWriter(object):
    def __init__(self, out_file_name):
        self.out_file_ = ROOT.TFile(out_file_name, "RECREATE")

    def fill(self, name, n_bins, v_min, v_max, values, weights=None):
        values = as_float_array(values)
        bins = np.linspace(v_min, v_max, n_bins + 1)
        h = ROOT.TH1D(name, name, len(bins) - 1, bins)
        if weights is None:
            weights = np.ones_like(values)
        weights = as_float_array(weights)
        h.FillN(len(values), values, weights)
        h.Write(name)

    def write(self):
        self.out_file_.Write()
