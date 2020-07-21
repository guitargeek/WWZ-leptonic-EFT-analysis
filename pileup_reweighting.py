import pandas as pd
import numpy as np


pileup_weights = {2017: pd.read_csv("resources/pileup_weigths/puw2017.csv")}


def get_pileup_weights(n_true_int, year):
    n_true_int = np.array(n_true_int, dtype=np.int)
    df_pu_weights = pileup_weights[year]

    weights = np.zeros(len(n_true_int), dtype=np.float)
    mask = np.logical_and(n_true_int >= 0, n_true_int < len(df_pu_weights))
    weights[mask] = df_pu_weights.loc[n_true_int[mask], "pu_weight_nominal"]

    return weights
