"""Effective-coordination summaries of a structure (55 values), as used for the training data.

Same definitions as the training descriptors: the cell is rebuilt from its lattice parameters,
neighbours are searched over the 27 nearest cell images within 8 A, and each neighbour gets the
weight exp[1 - (d/d_min)^6] (Hoppe's effective coordination number).
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

SEARCH_RADIUS = 8.0  # A
WEIGHT_EXPONENT = 6.0
CN_WEIGHT_THRESHOLD = 0.30
SELF_TOL = 1e-8
TRANSLATIONS = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float)
CN_HIST_BINS = list(range(1, 13))


def lattice_matrix(a, b, c, alpha_deg, beta_deg, gamma_deg):
    alpha, beta, gamma = (math.radians(x) for x in (alpha_deg, beta_deg, gamma_deg))
    va = np.array([a, 0.0, 0.0])
    vb = np.array([b * math.cos(gamma), b * math.sin(gamma), 0.0])
    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / math.sin(gamma)
    vc = np.array([cx, cy, math.sqrt(max(c * c - cx * cx - cy * cy, 0.0))])
    return np.vstack([va, vb, vc])


def site_descriptors(lattice, species, frac, radius=SEARCH_RADIUS):
    n = len(species)
    all_species = np.array(species, dtype=object)
    econ, cn, hetero, mean_dist = (np.zeros(n) for _ in range(4))
    for i in range(n):
        delta = frac[None, :, :] + TRANSLATIONS[:, None, :] - frac[i][None, None, :]
        dists = np.linalg.norm(delta @ lattice, axis=2)
        flat_d = dists.reshape(-1)
        flat_s = np.broadcast_to(all_species[None, :], dists.shape).reshape(-1)
        mask = (flat_d > SELF_TOL) & (flat_d <= radius)
        flat_d, flat_s = flat_d[mask], flat_s[mask]
        if flat_d.size == 0:
            raise ValueError("no neighbours within the search radius")
        w = np.exp(1.0 - (flat_d / flat_d.min()) ** WEIGHT_EXPONENT)
        econ[i] = w.sum()
        cn[i] = float(np.sum(w >= CN_WEIGHT_THRESHOLD))
        hetero[i] = float(w[flat_s != species[i]].sum() / econ[i]) if econ[i] > 0 else 0.0
        mean_dist[i] = float(np.dot(w, flat_d) / econ[i])
    return econ, cn, hetero, mean_dist


def _stats(arr, prefix):
    return {prefix + "_mean": float(np.mean(arr)), prefix + "_std": float(np.std(arr)),
            prefix + "_min": float(np.min(arr)), prefix + "_max": float(np.max(arr)),
            prefix + "_range": float(np.max(arr) - np.min(arr))}


def coordination_features(atoms) -> dict[str, float]:
    """55 summaries for an ase.Atoms (names as in the training data, without prefix)."""
    species = list(atoms.get_chemical_symbols())
    lattice = lattice_matrix(*atoms.cell.cellpar())
    frac = atoms.get_scaled_positions(wrap=False)
    econ, cn, hetero, dist = site_descriptors(lattice, species, frac)
    smap = defaultdict(list)
    for i, s in enumerate(species):
        smap[s].append(i)
    uniq = sorted(smap)
    out = {"nsites_struct": float(len(species)), "nelements_struct": float(len(uniq))}
    out.update(_stats(cn, "site_cn"))
    out.update(_stats(econ, "site_econ"))
    out.update(_stats(hetero, "site_hetero"))
    out.update(_stats(dist, "site_nn_dist"))
    out.update(_stats(np.array([np.mean(cn[smap[s]]) for s in uniq]), "species_cn"))
    out.update(_stats(np.array([np.mean(econ[smap[s]]) for s in uniq]), "species_econ"))
    out.update(_stats(np.array([np.mean(hetero[smap[s]]) for s in uniq]), "species_hetero"))
    out.update(_stats(np.array([np.mean(dist[smap[s]]) for s in uniq]), "species_nn_dist"))
    cnr = np.rint(cn).astype(int)
    counts = [float(np.mean(cnr == v)) for v in CN_HIST_BINS] + [float(np.mean(cnr >= 13))]
    for i, v in enumerate(counts, start=1):
        out["cn_hist_ge13" if i == len(counts) else f"cn_hist_{i}"] = v
    # WHY: round so that floating-point noise (a zero spread computed as 1e-16) cannot flip a tree split;
    # the training values are rounded the same way
    return {k: round(v, 8) for k, v in out.items()}
