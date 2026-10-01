"""Relax a structure with MACE-MPA-0 using the protocol of the training data.

Protocol (as recorded with the training relaxations): mace-mpa-0-medium, float64,
FixSymmetry(symprec 0.01), FrechetCellFilter (full cell + ions), FIRE, fmax 0.01 eV/A,
500 steps; a structure not converged by then is continued from its partially relaxed
cell for up to 3000 more steps. Requires the optional `mace` extra.
"""
from __future__ import annotations

import numpy as np

SYMPREC, FMAX, STAGES = 0.01, 0.01, (500, 3000)


def relax(atoms, device: str = "cpu", model: str = "medium-mpa-0") -> dict:
    """Returns the relaxation record in the field names of the training data."""
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE
    from ase.spacegroup.symmetrize import check_symmetry

    try:
        from ase.constraints import FixSymmetry
    except ImportError:
        from ase.spacegroup.symmetrize import FixSymmetry
    from mace.calculators import mace_mp

    atoms = atoms.copy()
    atoms.calc = mace_mp(model=model, device=device, default_dtype="float64")
    sg_init = int(check_symmetry(atoms, symprec=SYMPREC).number)
    atoms.set_constraint(FixSymmetry(atoms, symprec=SYMPREC))
    converged, steps = False, 0
    for max_steps in STAGES:
        flt = FrechetCellFilter(atoms)
        opt = FIRE(flt, logfile=None)
        converged = bool(opt.run(fmax=FMAX, steps=max_steps))
        steps += opt.get_number_of_steps()
        if converged:
            break
    n = len(atoms)
    stress = atoms.get_stress(voigt=True)
    a, b, c, al, be, ga = atoms.cell.cellpar()
    try:
        sg_final = int(check_symmetry(atoms, symprec=SYMPREC).number)
    except Exception:  # noqa: BLE001
        sg_final = None
    return {"V_final_per_atom": atoms.get_volume() / n,
            "a_final": a, "b_final": b, "c_final": c, "alpha_final": al, "beta_final": be, "gamma_final": ga,
            "E_final_per_atom": atoms.get_potential_energy() / n, "converged": converged, "n_steps": steps,
            "fmax_final": float(np.max(np.linalg.norm(flt.get_forces(), axis=1))),
            "pressure_final_eV_A3": float(-np.mean(stress[:3])),
            "max_abs_stress_eV_A3": float(np.max(np.abs(stress))),
            "sg_init": sg_init, "sg_final": sg_final, "atoms": atoms}


def features(fracs: dict[str, float], r: dict) -> dict[str, float]:
    """MACE-tier descriptors; must match training/build_dataset.py."""
    from volcorr import features as F

    x = F.geometry_features(fracs, r["V_final_per_atom"],
                            [r[f"{k}_final"] for k in ("a", "b", "c", "alpha", "beta", "gamma")], prefix="mace")
    sg_i, sg_f = r.get("sg_init"), r.get("sg_final")
    x.update({"mace_e_final_per_atom": float(r["E_final_per_atom"]),
              "mace_converged": float(bool(r["converged"])),
              "mace_fmax_final": float(r["fmax_final"]),
              "mace_pressure": float(r["pressure_final_eV_A3"]),
              "mace_max_abs_stress": float(r["max_abs_stress_eV_A3"]),
              "mace_sg_drift": np.nan if sg_i is None or sg_f is None else float(sg_i != sg_f)})
    return x
