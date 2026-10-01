"""User-facing entry point: pick the tier from what was given and return a Prediction."""
from __future__ import annotations

from volcorr import domain
from volcorr import features as F
from volcorr.model import Prediction, load

DFT_REF = "PBE / PBE+U, Materials Project relaxation settings"
MACE_REF = "MACE-MPA-0 (medium), symmetry-constrained relaxation"


def predict(formula: str | None = None, spacegroup: int | None = None, cif: str | None = None,
            v_dft: float | None = None, mace: bool = False, pbe_plus_u: bool | None = None,
            device: str = "cpu", structure_format: str | None = None, max_mace_atoms: int | None = None
            ) -> Prediction:
    """Predicted volume error.

    formula      e.g. "Fe2O3" (required unless a CIF is given)
    spacegroup   international number 1-230
    cif          structure file path or pasted contents (CIF, POSCAR, extxyz, ...)
    structure_format  ASE format name if auto-detection fails
    v_dft        your own PBE(+U) volume in A^3/atom -> corrected volume (tier D)
    mace         relax the CIF with MACE-MPA-0 and correct that volume (tier E)
    pbe_plus_u   override the Materials Project rule for Hubbard U (None = apply the rule)
    """
    atoms = F.read_structure(cif, structure_format) if cif else None
    if formula is None:
        if atoms is None:
            raise ValueError("give a formula or a CIF")
        formula = F.formula_of(atoms)
    fracs = F.parse_formula(formula)
    if atoms is not None:
        cif_fracs = F.parse_formula(F.formula_of(atoms))
        if set(cif_fracs) != set(fracs) or any(abs(cif_fracs[e] - fracs[e]) > 1e-3 for e in fracs):
            raise ValueError(f"formula {formula} does not match the CIF composition {F.formula_of(atoms)}")
        if spacegroup is None:
            spacegroup = F.spacegroup_of(atoms)

    x = {**F.composition_features(fracs), **F.shell_features(fracs),
         **F.settings_features(fracs, pbe_plus_u, energy_criterion=False)}
    if spacegroup is not None:
        x.update(F.symmetry_features(spacegroup))

    warnings: list[str] = []
    v_ref = v_in = None
    if mace:
        if atoms is None:
            raise ValueError("the MACE route needs a structure (CIF)")
        if max_mace_atoms and len(atoms) > max_mace_atoms:
            raise ValueError(f"the MACE route is limited to {max_mace_atoms} atoms here ({len(atoms)} given)")
        from volcorr import mace_route

        x.update(F.frac_features(fracs))
        rec = mace_route.relax(atoms, device=device)
        x.update(mace_route.features(fracs, rec))
        v_ref, tier, ref = rec["V_final_per_atom"], "E_mace", MACE_REF
        if not rec["converged"]:
            warnings.append("MACE relaxation did not converge; the corrected volume is less reliable.")
        if rec["sg_final"] != rec["sg_init"]:
            warnings.append(f"Space group changed during relaxation ({rec['sg_init']} -> {rec['sg_final']}).")
    elif v_dft is not None:
        if spacegroup is None:
            raise ValueError("the V_DFT route needs the space group (or a CIF)")
        x.update(F.geometry_features(fracs, float(v_dft), None, prefix="dft"))
        v_ref, tier, ref = float(v_dft), "D_vdft", DFT_REF
    elif atoms is not None:
        x.update(F.geometry_features(fracs, atoms.get_volume() / len(atoms), atoms.cell.cellpar()))
        v_in, tier, ref = atoms.get_volume() / len(atoms), "C_cif", DFT_REF
    elif spacegroup is not None:
        tier, ref = "B_spacegroup", DFT_REF
    else:
        tier, ref = "A_formula", DFT_REF

    m = load(tier)
    y_hat, ys = m.predict(x)
    i68, i90 = m.intervals(y_hat)
    warnings += domain.check(fracs, spacegroup, x)
    settings = {} if tier == "E_mace" else {
        "functional": "PBE+U" if x["set_pbe_plus_u"] else "PBE",
        "U_eV": {e: u for e, u in F._tables()[2]["U"].items() if e in fracs} if x["set_pbe_plus_u"] else {},
        "ENCUT_eV": 520, "spin_polarized": True, "ionic_stopping": "forces (EDIFFG = -0.05 eV/A)"}
    return Prediction(tier=tier, y_hat=y_hat, y_members=ys, reference=ref, interval68=i68, interval90=i90,
                      v_ref=v_ref, v_exp_input=v_in, warnings=warnings, settings=settings)
