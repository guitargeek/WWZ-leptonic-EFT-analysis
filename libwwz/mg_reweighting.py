import pandas as pd


def get_df_mg_reweighting_info(events):
    import io

    csv_str = events["LHEWeight_mg_reweighting"].title.decode("utf-8")
    return pd.read_csv(io.StringIO(csv_str), lineterminator=";", skiprows=1)


def get_df_mg_reweighting(events, standard_model=None):
    gen_weight = events.array("genWeight")

    df_info = get_df_mg_reweighting_info(events)
    mg_reweighting = gen_weight * events.array("LHEWeight_mg_reweighting")
    df = pd.DataFrame(mg_reweighting.flatten().reshape((len(mg_reweighting), len(mg_reweighting[0]))))
    df.columns = df_info["id"].values

    return df
    # if standard_model is None:
    # norm = np.sum(gen_weight)
    # else:
    # norm = df[standard_model].sum()

    # return df * xsec * lumi / norm
