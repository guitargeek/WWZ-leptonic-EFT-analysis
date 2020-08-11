import numpy as np
import pandas as pd

import awkward

from geeksw.utils.array_utils import unpack_pair_values, awksel


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
        # df_out[f"VetoLepton_passesZid_{i}"] = base_selection

    return df_out


def passes_W_id(df, use_z_id=False):
    if use_z_id:
        df_out = passes_Z_id(df)
        df_out.columns = [
            "VetoLepton_passesWid_0",
            "VetoLepton_passesWid_1",
            "VetoLepton_passesWid_2",
            "VetoLepton_passesWid_3",
        ]
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
        # df_out[f"VetoLepton_passesWid_{i}"] = base_selection

    return df_out


def get_complementary_indices(a, b, default=-99):

    c = np.zeros(len(a), dtype=np.int) + default
    d = np.zeros(len(b), dtype=np.int) + default

    c[np.logical_and(a == 0, b == 1)] = 2
    c[np.logical_and(a == 0, b == 2)] = 1
    c[np.logical_and(a == 0, b == 3)] = 1
    c[np.logical_and(a == 1, b == 2)] = 0
    c[np.logical_and(a == 1, b == 3)] = 0
    c[np.logical_and(a == 2, b == 3)] = 0

    d[np.logical_and(a == 0, b == 1)] = 3
    d[np.logical_and(a == 0, b == 2)] = 3
    d[np.logical_and(a == 0, b == 3)] = 2
    d[np.logical_and(a == 1, b == 2)] = 3
    d[np.logical_and(a == 1, b == 3)] = 2
    d[np.logical_and(a == 2, b == 3)] = 1

    return c, d


def pair_mass_array(df, prefix="VetoLeptonPair_mass_", jagged=True):
    return unpack_pair_values(df, column_getter=lambda i, j: prefix + f"{i}{j}", diagonal=np.nan, jagged=jagged)


def jagged_lepton_variable(df, variable):
    content = df[
        [f"VetoLepton_{variable}_0", f"VetoLepton_{variable}_1", f"VetoLepton_{variable}_2", f"VetoLepton_{variable}_3"]
    ].values.flatten()

    return awkward.JaggedArray.fromcounts(4 + np.zeros(len(df), dtype=np.int), content)


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

    z_df = pair_mass_array(find_z_candidates(df), prefix="VetoLeptonPair_z_cand_mass_", jagged=False).reshape(
        (len(df), -1)
    )

    has_z_cand = ~(np.sum(~np.isnan(z_df), axis=1) == 0)

    in_z_window = has_z_cand[:]
    in_z_window[has_z_cand] = np.nanmin(np.abs(z_df[has_z_cand] - z_mass), axis=1) < 10.0
    z_idx = np.nanargmin(np.abs(z_df[in_z_window] - z_mass), axis=1)

    z_lep_1_idx = np.zeros(len(df), dtype=np.int) - 99
    z_lep_2_idx = np.zeros(len(df), dtype=np.int) - 99

    z_lep_1_idx[in_z_window] = z_idx // 4
    z_lep_2_idx[in_z_window] = z_idx % 4

    w_lep_1_idx, w_lep_2_idx = get_complementary_indices(z_lep_1_idx, z_lep_2_idx)

    # The invariant mass of the W-candidate leptons no matter if they make a Z-boson or not
    m_ll = np.zeros(len(df), dtype=np.int)
    pair_masses = pair_mass_array(df)
    m_ll[in_z_window] = awksel(pair_masses, [w_lep_1_idx, w_lep_2_idx], mask=in_z_window)

    # Don't consider W-leptons if they don' pass the ID
    passes_w_id = jagged_lepton_variable(df, "passesWid")[in_z_window]
    w_lep_1_idx_in_z_window = w_lep_1_idx[in_z_window].copy()
    w_lep_2_idx_in_z_window = w_lep_2_idx[in_z_window].copy()
    passes = np.logical_and(
        awksel(passes_w_id, [w_lep_1_idx_in_z_window]), awksel(passes_w_id, [w_lep_2_idx_in_z_window])
    ).flatten()
    passes_all = np.array(in_z_window)
    passes_all[in_z_window] = passes
    w_lep_1_idx[~passes_all] = -99
    w_lep_2_idx[~passes_all] = -99

    # Don't consider W-leptons if they are not opposite charge
    w_lep_charge = np.sign(jagged_lepton_variable(df, "pdgId"))[in_z_window]
    w_charge_1 = awksel(w_lep_charge, [w_lep_1_idx_in_z_window])
    w_charge_2 = awksel(w_lep_charge, [w_lep_2_idx_in_z_window])
    passes = w_charge_1 + w_charge_2 == 0
    passes_all = np.array(in_z_window)
    passes_all[in_z_window] = passes
    w_lep_1_idx[~passes_all] = -99
    w_lep_2_idx[~passes_all] = -99

    # W candidate pt cuts
    w_cand_pt = jagged_lepton_variable(df, "pt")[in_z_window]
    w_pt_1 = awksel(w_cand_pt, [w_lep_1_idx_in_z_window])
    w_pt_2 = awksel(w_cand_pt, [w_lep_2_idx_in_z_window])
    passes = np.logical_and(np.max([w_pt_1, w_pt_2], axis=0) > 25, np.min([w_pt_1, w_pt_2], axis=0) > 10)

    passes_all = np.array(in_z_window)
    passes_all[in_z_window] = passes
    w_lep_1_idx[~passes_all] = -99
    w_lep_2_idx[~passes_all] = -99

    return pd.DataFrame(
        dict(
            z_lep_1_idx=z_lep_1_idx,
            z_lep_2_idx=z_lep_2_idx,
            w_lep_1_idx=w_lep_1_idx,
            w_lep_2_idx=w_lep_2_idx,
            m_ll=m_ll,
        )
    )


def get_z_cand_masses(df):
    has_z_cand_1 = df["z_lep_1_idx"] >= 0
    has_z_cand_2 = df["w_lep_1_idx"] >= 0

    z_lep_1_idx = df["z_lep_1_idx"][has_z_cand_1].values
    z_lep_2_idx = df["z_lep_2_idx"][has_z_cand_1].values
    w_lep_1_idx = df["w_lep_1_idx"][has_z_cand_2].values
    w_lep_2_idx = df["w_lep_2_idx"][has_z_cand_2].values

    pair_masses = pair_mass_array(df)

    z_cand_1_mass = awksel(pair_masses[has_z_cand_1], [z_lep_1_idx, z_lep_2_idx])
    z_cand_2_mass = awksel(pair_masses[has_z_cand_2], [w_lep_1_idx, w_lep_2_idx])

    lep_pdg_id = jagged_lepton_variable(df, "pdgId")[has_z_cand_2]
    w_pdg_1 = awksel(lep_pdg_id, [w_lep_1_idx])
    w_pdg_2 = awksel(lep_pdg_id, [w_lep_2_idx])

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
