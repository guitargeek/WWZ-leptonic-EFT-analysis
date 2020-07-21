import numpy as np
import awkward

from libwwz.array_utils import awkward_indices


def find_z_particle_and_antiparticle(particles, antiparticles, z_mass=91.1876):
    combinations = particles.cross(antiparticles, nested=False)
    pair_masses = (combinations.i0 + combinations.i1).mass

    pair_indices = awkward_indices(pair_masses)

    anti_counts = antiparticles.counts
    particle_idx = pair_indices // anti_counts
    antiparticle_idx = pair_indices % anti_counts

    pair_residuals = np.abs(pair_masses - z_mass)
    argmins = pair_residuals.argmin()

    return particle_idx[argmins], antiparticle_idx[argmins], pair_masses[argmins].min()


def find_z_pairs_for_flavor(p4, lep_id, pdg_id, z_mass=91.1876):

    pdg_id = abs(pdg_id)

    particle_idx, antiparticle_idx, masses = find_z_particle_and_antiparticle(
        p4[lep_id == -pdg_id], p4[lep_id == pdg_id], z_mass=z_mass
    )

    mask = awkward.JaggedArray.fromcounts(p4.counts, np.zeros(len(p4.flatten()), dtype=np.bool))

    lep_idx = awkward_indices(p4)

    mask[lep_idx[lep_id == -pdg_id][particle_idx]] = True
    mask[lep_idx[lep_id == pdg_id][antiparticle_idx]] = True

    return mask, masses


def find_z_pairs(p4, lep_id, z_mass=91.1876, z_window=10.0):
    ele_pair_mask, ele_pair_mass = find_z_pairs_for_flavor(p4, lep_id, 11, z_mass=z_mass)
    mu_pair_mask, mu_pair_mass = find_z_pairs_for_flavor(p4, lep_id, 13, z_mass=z_mass)

    mu_is_better = np.abs(mu_pair_mass - z_mass) < np.abs(ele_pair_mass - z_mass)

    lep_pair_mask = mu_pair_mask & mu_is_better | ele_pair_mask & ~mu_is_better

    lep_pair_mass = ele_pair_mass.copy()
    lep_pair_mass[mu_is_better] = mu_pair_mass[mu_is_better]

    is_in_z_window = np.abs(lep_pair_mass - z_mass) <= z_window

    lep_pair_mass[~is_in_z_window] = np.inf
    lep_pair_mask = lep_pair_mask & is_in_z_window

    lep_pair_mass[lep_pair_mass == np.inf] = np.nan

    return lep_pair_mask
