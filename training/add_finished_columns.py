"""Add descriptors of a finished PBE(+U) calculation to _private/dataset.npz (same rows).

fs_*  relaxed cell angles and axis ratios      (from the MP relaxed structure)
fg_*  band gap, log(1+gap), gap > 0.1 eV       (MP band gap)
fm_*  |magnetization| per volume and per atom, magnetic flag (MP summary)
fc_*  effective-coordination summaries         (relaxed structure, volcorr.coordination definitions)
fe_*  formation energy and energy above hull   (MP summary; optional inputs)
Values are taken from the recomputed descriptor matrix of study_delta10 (identical to the published
matrix for these entries).
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "study_delta10" / "data" / "features_dft_hybrid.npz"
D = np.load(HERE / "_private" / "dataset.npz", allow_pickle=True)
F = np.load(SRC, allow_pickle=True)
fn = [str(x) for x in F["feature_names"]]
pos = {str(m): i for i, m in enumerate(F["mp_ids"])}
rows = np.array([pos[str(m)] for m in D["mp_ids"]])
X = F["X"][rows].astype(np.float64)
f = lambda c: X[:, fn.index(c)]
vpa = f("vpa")
mag_vol = np.abs(f("elec_total_magnetization_per_vol"))  # mu_B / A^3, independent of the cell choice
new = {"fs_alpha": f("alpha"), "fs_beta": f("beta"), "fs_gamma": f("gamma"), "fs_a_over_b": f("a_over_b"),
       "fs_a_over_c": f("a_over_c"), "fs_b_over_c": f("b_over_c"),
       "fg_gap": f("post_dft_band_gap_ev"), "fg_log1p_gap": f("post_dft_log1p_band_gap_ev"),
       "fg_gapped": f("post_dft_is_gapped_gt_0p1ev"),
       "fm_mag_vol": mag_vol, "fm_mag_atom": mag_vol * vpa, "fm_magnetic": (mag_vol * vpa > 0.05).astype(float),
       "fe_formation_energy": f("elec_formation_energy_per_atom"), "fe_e_above_hull": f("elec_e_above_hull")}
c0 = fn.index("nsites_struct")
for j in range(c0, c0 + 55):
    # WHY: rounded as in volcorr.coordination, so that round-off (e.g. 1e-16 for a zero spread) cannot
    # decide a tree split; the same rounding is applied at prediction time
    new["fc_" + fn[j]] = np.round(X[:, j], 8)
cols = [str(c) for c in D["columns"]]
keep = [i for i, c in enumerate(cols) if not c.startswith(("fs_", "fg_", "fm_", "fc_", "fe_"))]
out = {k: D[k] for k in D.files}
out["X"] = np.hstack([D["X"][:, keep], np.column_stack(list(new.values()))])
out["columns"] = np.array([cols[i] for i in keep] + list(new))
np.savez_compressed(HERE / "_private" / "dataset.npz", **out)
print("dataset:", out["X"].shape)
