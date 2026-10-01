"""Assemble the training table with the package featurizer (ICSD-derived; stays private).

Rows follow the paper's split files (29,413 MP-ICSD entries with |delta_DFT| <= 5%); `keep`
reproduces the 29,372 benchmark rows. Columns:

  comp_*, shell_*, sym_*   formula and space group (space group of the input CIF, symprec 0.01)
  set_*                    recorded MP settings (functional, U values, EDIFFG group)
  rule_set_*               the same settings as the MP rule would assign them (inference view)
  cell_*                   experimental input cell
  dft_*                    DFT volume scalars (tier D: the user supplies V_DFT)
  mace_*                   MACE-MPA-0 relaxation of the input cell (paper protocol)

Output: training/_private/dataset.npz
"""
from __future__ import annotations

import glob
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from volcorr import features as F  # noqa: E402

REPO = Path("/nobackup/proj/disk/snic2014-8-11/personal/usingh/alvis/corrections/mp_earliest_relaxed_icsd")
MACE = Path("/nobackup/proj/disk/snic2014-8-11/personal/usingh/alvis/corrections/ml_on_v/mace_study")
OUT = HERE / "_private" / "dataset.npz"


def load_mace(pattern: str) -> dict:
    """mp_id -> record; converged first, then smallest residual force (paper's rule)."""
    rows = {}
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not d.get("mp_id") or "error" in d or d.get("V_final_per_atom") is None:
                continue
            key = lambda r: (bool(r.get("converged")), -float(r.get("fmax_final", np.inf)))
            if d["mp_id"] not in rows or key(d) > key(rows[d["mp_id"]]):
                rows[d["mp_id"]] = d
    return rows


def featurize(args):
    formula, sg, pbeu, energy_crit, cif, v_dft, mrow = args
    fr = F.parse_formula(formula)
    atoms = None
    try:
        atoms = F.read_structure(cif)
        sg = F.spacegroup_of(atoms)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] {cif}: {type(e).__name__}: {e}", flush=True)
    x = {**F.composition_features(fr), **F.shell_features(fr),
         **F.settings_features(fr, pbeu, energy_crit)}
    x.update({f"rule_{k}": v for k, v in F.settings_features(fr, None, False).items()})
    if sg is not None:
        x.update(F.symmetry_features(sg))
    if atoms is not None:
        x.update(F.geometry_features(fr, atoms.get_volume() / len(atoms), atoms.cell.cellpar(), prefix="cell"))
    x.update(F.geometry_features(fr, v_dft, None, prefix="dft"))
    if mrow is not None:
        x.update(F.geometry_features(fr, mrow["V_final_per_atom"],
                                     [mrow[f"{k}_final"] for k in ("a", "b", "c", "alpha", "beta", "gamma")],
                                     prefix="mace"))
        x.update(mace_outputs(mrow))
    return x


def mace_outputs(r: dict) -> dict:
    """Start-independent MACE relaxation record (the step count and energy drop are left out)."""
    sg_i, sg_f = r.get("sg_init"), r.get("sg_final")
    return {"mace_e_final_per_atom": float(r["E_final_per_atom"]),
            "mace_converged": float(bool(r.get("converged"))),
            "mace_fmax_final": float(r.get("fmax_final", np.nan)),
            "mace_pressure": float(r.get("pressure_final_eV_A3", np.nan)),  # eV/A^3
            "mace_max_abs_stress": float(r.get("max_abs_stress_eV_A3", np.nan)),
            "mace_sg_drift": np.nan if sg_i is None or sg_f is None else float(sg_i != sg_f)}


def main() -> None:
    sp = np.load(REPO / "study_delta5_gga_u_v3/data/splits_delta5_pbe_pbeu.npz", allow_pickle=True)
    mp_ids = np.array([str(m) for m in sp["mp_ids"]])
    Fm = np.load(MACE / "data/mace_mace_mpa_0_features.npz", allow_pickle=True)
    Vm = np.load(MACE / "data/mace_mace_mpa_0_volumes.npz", allow_pickle=True)
    assert (np.array([str(m) for m in Fm["mp_ids"]]) == mp_ids).all()
    assert (np.array([str(m) for m in Vm["mp_ids"]]) == mp_ids).all()
    names = [str(x) for x in Fm["feature_names"]]
    rec = lambda c: Fm["X"][:, names.index(c)]
    pbeu = rec("calc_is_pbeu") > 0.5
    energy_crit = rec("calc_ediffg_is_force") < 0.5
    sg_fallback = rec("spacegroup_num")

    csv = pd.read_csv(REPO / "mp_icsd_dataset.csv", keep_default_na=False).set_index("mp_id")
    formulas = csv.loc[mp_ids, "formula"].astype(str).to_numpy()
    v_exp, v_dft, v_mace = Vm["v_exp"], Vm["v_dft"], Vm["v_mace"]
    mrows = load_mace(str(MACE / "results/mace_mpa_0/*.jsonl"))
    print(f"MACE records: {len(mrows):,}")

    jobs = []
    for i, m in enumerate(mp_ids):
        r = mrows.get(m)
        sg = int(sg_fallback[i]) if np.isfinite(sg_fallback[i]) else None  # used only if the CIF fails
        cif = str(REPO / csv.loc[m, "initial_cif_path"])
        jobs.append((formulas[i], sg, bool(pbeu[i]), bool(energy_crit[i]), cif, float(v_dft[i]), r))
    with ProcessPoolExecutor(32) as ex:
        feats = list(ex.map(featurize, jobs, chunksize=64))
    cols = sorted({k for x in feats for k in x}, key=lambda k: (k.split("_")[0], k))
    X = np.array([[x.get(c, np.nan) for c in cols] for x in feats], dtype=np.float64)

    keep = Fm["valid_mask"].astype(bool) & np.isfinite(v_exp) & np.isfinite(v_dft)
    # WHY: the input cell is the experimental structure, so its volume must equal V_exp.
    cv = X[:, cols.index("cell_volume_per_atom")]
    ok = np.isfinite(cv)
    print(f"input-cell volume vs V_exp: max rel diff {np.nanmax(np.abs(cv[ok] / v_exp[ok] - 1)):.2e} "
          f"({ok.sum():,} rows with a cell)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, X=X, columns=np.array(cols), mp_ids=mp_ids, formulas=formulas,
                        keep=keep, v_exp=v_exp, v_dft=v_dft, v_mace=v_mace,
                        pbeu_recorded=pbeu, energy_criterion=energy_crit)
    print(f"wrote {OUT}: {X.shape}, keep={keep.sum():,}")


if __name__ == "__main__":
    main()
