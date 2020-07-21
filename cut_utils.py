import numpy as np


def add_cut(cut_df, cut_name, cut_result, previous_cut=None, weight=1.0):

    if previous_cut is None:
        cut_df[cut_name] = cut_result
    else:
        cut_df[cut_name] = np.logical_and(cut_df[previous_cut], cut_result)

    weight = weight * cut_df[cut_name]

    if previous_cut is None:
        cut_df[cut_name + "Weight"] = weight
    else:
        cut_df[cut_name + "Weight"] = weight * cut_df[previous_cut + "Weight"]

    return cut_df
