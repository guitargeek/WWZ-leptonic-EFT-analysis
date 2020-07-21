import awkward
import numpy as np


def reduce_and(*arrays):
    if isinstance(arrays[0], np.ndarray):
        return np.logical_and.reduce(arrays)
    else:
        result = np.logical_and.reduce([a.flatten() for a in arrays])
        return awkward.JaggedArray(arrays[0].starts, arrays[0].stops, result)


def reduce_or(*arrays):
    if isinstance(arrays[0], np.ndarray):
        return np.logical_or.reduce(arrays)
    else:
        result = np.logical_or.reduce([a.flatten() for a in arrays])
        return awkward.JaggedArray(arrays[0].starts, arrays[0].stops, result)


def awkward_indices(array):
    counts = array.counts
    idx = awkward.JaggedArray.fromcounts(counts, np.arange(len(array.flatten())))
    idx = idx - np.cumsum(counts) + counts
    return idx
