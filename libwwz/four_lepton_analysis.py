import numpy as np
import pandas as pd

import awkward


def passes_Z_id(df):
    df_out = pd.DataFrame()
    for i in range(4):
        base_selection = df[f"VetoLepton_ip3d_{i}"].abs() / df[f"VetoLepton_sip3d_{i}"] < 4

        is_ele = df[f"VetoLepton_pdgId_{i}"].abs() == 11
        is_mu = df[f"VetoLepton_pdgId_{i}"].abs() == 13

        electron_selection = df[f"VetoLepton_pfRelIso03_all_wLep_{i}"] < 0.2

        muon_selection = df[f"VetoLepton_mediumId_{i}"] & df[f"VetoLepton_pfRelIso03_all_wLep_{i}"] < 0.25

        selection = base_selection & ((is_ele & electron_selection) | (is_mu & muon_selection))

        df_out[f"VetoLepton_passesZid_{i}"] = selection
    return df_out


def passes_W_id(df, use_z_id=False):
    if use_z_id:
        df_out = passes_Z_id(df)
        df_out.columns = ["VetoLepton_passesWid_0", "VetoLepton_passesWid_1", "VetoLepton_passesWid_2", "VetoLepton_passesWid_3"]
        return df_out
    df_out = pd.DataFrame()
    for i in range(4):
        base_selection = df[f"VetoLepton_ip3d_{i}"].abs() / df[f"VetoLepton_sip3d_{i}"] < 4

        is_ele = df[f"VetoLepton_pdgId_{i}"].abs() == 11
        is_mu = df[f"VetoLepton_pdgId_{i}"].abs() == 13

        electron_selection = df[f"VetoLepton_mediumId_{i}"] & df[f"VetoLepton_pfRelIso03_all_wLep_{i}"] < 0.2

        muon_selection = df[f"VetoLepton_mediumId_{i}"] & df[f"VetoLepton_pfRelIso03_all_wLep_{i}"] < 0.15

        selection = base_selection & ((is_ele & electron_selection) | (is_mu & muon_selection))

        df_out[f"VetoLepton_passesWid_{i}"] = selection
    return df_out


def jagged_pair_masses(df):
    content = df[
        [
            "VetoLeptonPair_mass_01",
            "VetoLeptonPair_mass_02",
            "VetoLeptonPair_mass_03",
            "VetoLeptonPair_mass_12",
            "VetoLeptonPair_mass_13",
            "VetoLeptonPair_mass_23",
        ]
    ].values.flatten()

    return awkward.JaggedArray.fromcounts(6 + np.zeros(len(df), dtype=np.int), content)


def jagged_lepton_variable(df, variable):
    content = df[
        [f"VetoLepton_{variable}_0", f"VetoLepton_{variable}_1", f"VetoLepton_{variable}_2", f"VetoLepton_{variable}_3"]
    ].values.flatten()

    return awkward.JaggedArray.fromcounts(4 + np.zeros(len(df), dtype=np.int), content)


def to_singleton_jagged_array(arr):
    return awkward.JaggedArray.fromcounts(np.ones(len(arr), dtype=np.int), arr)


def find_boson_candidate_indices(df):
    z_mass = 91.19

    def find_z_candidates(df):

        df_out = pd.DataFrame()

        for i in range(4):
            for j in range(4):
                if j <= i:
                    continue
                mass = df[f"VetoLeptonPair_mass_{i}{j}"].copy().values
                mass[df[f"VetoLepton_pdgId_{i}"] + df[f"VetoLepton_pdgId_{j}"] != 0] = np.nan
                mass[~df[f"VetoLepton_passesZid_{i}"]] = np.nan
                mass[~df[f"VetoLepton_passesZid_{j}"]] = np.nan

                pt_i = df[f"VetoLepton_pt_{i}"]
                pt_j = df[f"VetoLepton_pt_{j}"]
                mass[np.max([pt_i, pt_j], axis=0) < 25] = np.nan
                mass[np.min([pt_i, pt_j], axis=0) < 10] = np.nan

                df_out[f"VetoLeptonPair_z_cand_mass_{i}{j}"] = mass

        return df_out

    z_df = find_z_candidates(df)

    def z_idx_to_lep_idx(z_idx):
        lep_1_idx = 0 + (z_idx > 2)
        lep_2_idx = z_idx + 1

        lep_1_idx = lep_1_idx + (z_idx > 4)

        lep_2_idx[z_idx == 2] = 3
        lep_2_idx[z_idx == 3] = 2
        lep_2_idx[z_idx == 4] = 3
        lep_2_idx[z_idx == 5] = 3

        lep_1_idx[z_idx < 0] = -99
        lep_2_idx[z_idx < 0] = -99

        return lep_1_idx, lep_2_idx

    has_z_cand = ~(np.sum(~np.isnan(z_df.values), axis=1) == 0)

    in_z_window = has_z_cand[:]
    in_z_window[has_z_cand] = np.nanmin(np.abs(z_df[has_z_cand].values - z_mass), axis=1) < 10.0
    z_idx = np.nanargmin(np.abs(z_df[in_z_window].values - z_mass), axis=1)

    w_lep_1_idx = np.zeros(len(df), dtype=np.int) - 99
    w_lep_2_idx = np.zeros(len(df), dtype=np.int) - 99

    z_lep_1_idx = np.zeros(len(df), dtype=np.int) - 99
    z_lep_2_idx = np.zeros(len(df), dtype=np.int) - 99

    a, b = z_idx_to_lep_idx(z_idx)
    z_lep_1_idx[in_z_window] = a
    z_lep_2_idx[in_z_window] = b

    w_lep_1_idx[np.logical_and(z_lep_1_idx == 0, z_lep_2_idx == 1)] = 2
    w_lep_1_idx[np.logical_and(z_lep_1_idx == 0, z_lep_2_idx == 2)] = 1
    w_lep_1_idx[np.logical_and(z_lep_1_idx == 0, z_lep_2_idx == 3)] = 1
    w_lep_1_idx[np.logical_and(z_lep_1_idx == 1, z_lep_2_idx == 2)] = 0
    w_lep_1_idx[np.logical_and(z_lep_1_idx == 1, z_lep_2_idx == 3)] = 0
    w_lep_1_idx[np.logical_and(z_lep_1_idx == 2, z_lep_2_idx == 3)] = 0

    w_lep_2_idx[np.logical_and(z_lep_1_idx == 0, z_lep_2_idx == 1)] = 3
    w_lep_2_idx[np.logical_and(z_lep_1_idx == 0, z_lep_2_idx == 2)] = 3
    w_lep_2_idx[np.logical_and(z_lep_1_idx == 0, z_lep_2_idx == 3)] = 2
    w_lep_2_idx[np.logical_and(z_lep_1_idx == 1, z_lep_2_idx == 2)] = 3
    w_lep_2_idx[np.logical_and(z_lep_1_idx == 1, z_lep_2_idx == 3)] = 2
    w_lep_2_idx[np.logical_and(z_lep_1_idx == 2, z_lep_2_idx == 3)] = 1

    # Don't consider W-leptons if they don' pass the ID
    passes_w_id = jagged_lepton_variable(df, "passesWid")[in_z_window]
    jagged_w_lep_1_idx_in_z_window = to_singleton_jagged_array(w_lep_1_idx[in_z_window])
    jagged_w_lep_2_idx_in_z_window = to_singleton_jagged_array(w_lep_2_idx[in_z_window])
    passes = np.logical_and(
        passes_w_id[jagged_w_lep_1_idx_in_z_window], passes_w_id[jagged_w_lep_2_idx_in_z_window]
    ).flatten()
    passes_all = np.array(in_z_window)
    passes_all[in_z_window] = passes
    w_lep_1_idx[~passes_all] = -99
    w_lep_2_idx[~passes_all] = -99

    # Don't consider W-leptons if they are not opposite charge
    w_lep_charge = np.sign(jagged_lepton_variable(df, "pdgId"))[in_z_window]
    w_charge_1 = w_lep_charge[jagged_w_lep_1_idx_in_z_window].flatten()
    w_charge_2 = w_lep_charge[jagged_w_lep_2_idx_in_z_window].flatten()
    passes = w_charge_1 + w_charge_2 == 0
    passes_all = np.array(in_z_window)
    passes_all[in_z_window] = passes
    w_lep_1_idx[~passes_all] = -99
    w_lep_2_idx[~passes_all] = -99

    # W candidate pt cuts
    w_cand_pt = jagged_lepton_variable(df, "pt")[in_z_window]
    w_pt_1 = w_cand_pt[jagged_w_lep_1_idx_in_z_window].flatten()
    w_pt_2 = w_cand_pt[jagged_w_lep_2_idx_in_z_window].flatten()
    passes = np.logical_and(np.max([w_pt_1, w_pt_2], axis=0) > 25, np.min([w_pt_1, w_pt_2], axis=0) > 10)

    passes_all = np.array(in_z_window)
    passes_all[in_z_window] = passes
    w_lep_1_idx[~passes_all] = -99
    w_lep_2_idx[~passes_all] = -99

    return pd.DataFrame(
        dict(z_lep_1_idx=z_lep_1_idx, z_lep_2_idx=z_lep_2_idx, w_lep_1_idx=w_lep_1_idx, w_lep_2_idx=w_lep_2_idx)
    )


def lep_idx_to_z_idx(z_lep_1_idx, z_lep_2_idx):
    z_idx = np.zeros(len(z_lep_1_idx), dtype=np.int) - 99

    z_idx[np.logical_and(z_lep_1_idx == 0, z_lep_2_idx == 1)] = 0
    z_idx[np.logical_and(z_lep_1_idx == 0, z_lep_2_idx == 2)] = 1
    z_idx[np.logical_and(z_lep_1_idx == 0, z_lep_2_idx == 3)] = 2
    z_idx[np.logical_and(z_lep_1_idx == 1, z_lep_2_idx == 2)] = 3
    z_idx[np.logical_and(z_lep_1_idx == 1, z_lep_2_idx == 3)] = 4
    z_idx[np.logical_and(z_lep_1_idx == 2, z_lep_2_idx == 3)] = 5

    return z_idx


def get_z_cand_masses(df):
    has_z_cand_1 = df["z_lep_1_idx"] >= 0
    has_z_cand_2 = df["w_lep_1_idx"] >= 0

    z_lep_1_idx = df["z_lep_1_idx"][has_z_cand_1].values
    z_lep_2_idx = df["z_lep_2_idx"][has_z_cand_1].values
    w_lep_1_idx = df["w_lep_1_idx"][has_z_cand_2].values
    w_lep_2_idx = df["w_lep_2_idx"][has_z_cand_2].values

    pair_masses = jagged_pair_masses(df)

    z_cand_1_mass = pair_masses[has_z_cand_1][to_singleton_jagged_array(lep_idx_to_z_idx(z_lep_1_idx, z_lep_2_idx))]
    z_cand_2_mass = pair_masses[has_z_cand_2][to_singleton_jagged_array(lep_idx_to_z_idx(w_lep_1_idx, w_lep_2_idx))]

    lep_pdg_id = jagged_lepton_variable(df, "pdgId")[has_z_cand_2]
    w_pdg_1 = lep_pdg_id[to_singleton_jagged_array(w_lep_1_idx)]
    w_pdg_2 = lep_pdg_id[to_singleton_jagged_array(w_lep_2_idx)]

    z_cand_2_mass[w_pdg_1 + w_pdg_2 != 0] = np.nan

    df_out = pd.DataFrame(index=df.index)
    df_out.loc[has_z_cand_1, "ZCand_mass_0"] = z_cand_1_mass
    df_out.loc[has_z_cand_2, "ZCand_mass_1"] = z_cand_2_mass
    return df_out


def is_emu_category(df):
    selection = np.sum(df[["z_lep_1_idx", "z_lep_2_idx", "w_lep_1_idx", "w_lep_2_idx"]].values, axis=1) == 6
    selection = np.logical_and(selection, df["ZCand_mass_1"].isna())
    return np.logical_and(selection, df["nb"] == 0)


def is_btag_emu_category(df):
    selection = np.sum(df[["z_lep_1_idx", "z_lep_2_idx", "w_lep_1_idx", "w_lep_2_idx"]].values, axis=1) == 6
    selection = np.logical_and(selection, df["ZCand_mass_1"].isna())
    return np.logical_and(selection, df["nb"] > 0)


def is_offz_category(df):
    selection = np.sum(df[["z_lep_1_idx", "z_lep_2_idx", "w_lep_1_idx", "w_lep_2_idx"]].values, axis=1) == 6
    selection = np.logical_and(selection, (df["ZCand_mass_1"] - 91.19).abs() > 10.0)
    return selection


def is_onz_category(df):
    selection = np.sum(df[["z_lep_1_idx", "z_lep_2_idx", "w_lep_1_idx", "w_lep_2_idx"]].values, axis=1) == 6
    selection = np.logical_and(selection, (df["ZCand_mass_1"] - 91.19).abs() <= 10.0)
    return selection


def four_lepton_analysis(df, use_z_id_as_w_id=False):
    df = df[:].copy()
    df = pd.concat([df, passes_Z_id(df), passes_W_id(df, use_z_id=use_z_id_as_w_id)], axis=1)
    df_cands_idx = find_boson_candidate_indices(df)
    df = pd.concat([df, df_cands_idx], axis=1)
    df_z_cand_masses = get_z_cand_masses(df)
    df = pd.concat([df, df_z_cand_masses], axis=1)
    df["category"] = "uncategorized"
    df.loc[is_btag_emu_category(df), "category"] = "BTagEMu"
    df.loc[is_emu_category(df), "category"] = "EMu"
    df.loc[is_offz_category(df), "category"] = "OffZ"
    df.loc[is_onz_category(df), "category"] = "OnZ"
    df["veto_lepton_pt_sum"] = jagged_lepton_variable(df, "pt").sum()
    df["veto_lepton_pt_min"] = jagged_lepton_variable(df, "pt").min()
    return df
