"""Descriptors computed from a formula, a space group or a crystal structure.

The same functions build the training matrix and the inputs at prediction time, so the
two cannot drift apart. Nothing here needs a DFT calculation, and nothing needs pymatgen:
element data come from bundled tables (training/make_tables.py), so the featurizer also
runs in the browser (Pyodide).
"""
from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from importlib import resources

import numpy as np

STATS = ("mean", "avg_dev", "minimum", "maximum", "range", "mode")


@lru_cache(maxsize=1)
def _tables():
    data = resources.files("volcorr") / "data"
    el = json.loads((data / "elements.json").read_text())
    sg = json.loads((data / "spacegroups.json").read_text())
    mpu = json.loads((data / "mp_u.json").read_text())
    return el, sg, mpu


@lru_cache(maxsize=1)
def basics() -> dict:
    """Per element: Z, X (Pauling), mass (amu), radius (A), is_tm, n_d, n_f."""
    return json.loads((resources.files("volcorr") / "data" / "element_basics.json").read_text())


_TOKEN = re.compile(r"([A-Z][a-z]?)|(\d*\.\d+|\d+)|([(\[{])|([)\]}])")


def _parse_counts(formula: str) -> dict[str, float]:
    """Element -> amount; nested (), [], {} with multipliers and hydrates (CuSO4·5H2O, CuSO4*5H2O)."""
    # WHY: "." is always a decimal point (Li0.5CoO2); hydrates need "·" or "*" (CuSO4·5H2O).
    parts = re.split(r"\s*[·•*]\s*", formula.strip())
    total: dict[str, float] = {}
    for part in parts:
        m = re.match(r"^(\d*\.?\d+)(?=[A-Z(\[{])", part)
        mult = float(m.group(1)) if m else 1.0
        body = part[m.end():] if m else part
        stack: list[dict[str, float]] = [{}]
        pos, last = 0, None  # last: ("el", sym) or ("group", dict) awaiting a count
        body = body.replace(" ", "")
        while pos < len(body):
            t = _TOKEN.match(body, pos)
            if t is None:
                raise ValueError(f"cannot parse formula {formula!r} at {body[pos:]!r}")
            pos = t.end()
            el, num, op, _ = t.groups()
            if num is not None:
                if last is None:
                    raise ValueError(f"misplaced number in formula {formula!r}")
                kind, val = last
                if kind == "el":
                    stack[-1][val] = stack[-1].get(val, 0.0) + float(num) - 1.0
                else:
                    for k, v in val.items():
                        stack[-1][k] = stack[-1].get(k, 0.0) + v * (float(num) - 1.0)
                last = None
            elif el is not None:
                if el not in basics():
                    raise ValueError(f"unknown element {el!r} in formula {formula!r}")
                stack[-1][el] = stack[-1].get(el, 0.0) + 1.0
                last = ("el", el)
            elif op is not None:
                stack.append({})
                last = None
            else:
                if len(stack) == 1:
                    raise ValueError(f"unbalanced brackets in formula {formula!r}")
                grp = stack.pop()
                for k, v in grp.items():
                    stack[-1][k] = stack[-1].get(k, 0.0) + v
                last = ("group", grp)
        if len(stack) != 1:
            raise ValueError(f"unbalanced brackets in formula {formula!r}")
        for k, v in stack[0].items():
            total[k] = total.get(k, 0.0) + mult * v
    return {k: v for k, v in total.items() if v > 1e-8}


def parse_formula(formula: str) -> dict[str, float]:
    """Element -> atomic fraction."""
    amounts = _parse_counts(formula)
    tot = sum(amounts.values())
    if tot <= 0:
        raise ValueError(f"empty formula: {formula!r}")
    return {el: amt / tot for el, amt in amounts.items()}


def _weighted_stats(values: np.ndarray, weights: np.ndarray) -> list[float]:
    ok = ~np.isnan(values)
    if not ok.any():
        return [math.nan] * len(STATS)
    v, w = values[ok], weights[ok] / weights[ok].sum()
    mean = float((v * w).sum())
    # WHY: mode = value of the most abundant element; ties resolved by the smaller value,
    # which makes the descriptor independent of element order.
    top = w.max()
    mode = float(v[np.isclose(w, top)].min())
    return [mean, float((np.abs(v - mean) * w).sum()), float(v.min()), float(v.max()),
            float(v.max() - v.min()), mode]


def composition_features(fracs: dict[str, float]) -> dict[str, float]:
    el_tab, _, _ = _tables()
    props = el_tab["properties"]
    syms = list(fracs)
    w = np.array([fracs[s] for s in syms])
    out: dict[str, float] = {}
    for p in props:
        vals = np.array([np.nan if el_tab["elements"][s][p] is None else el_tab["elements"][s][p]
                         for s in syms], dtype=float)
        for stat, v in zip(STATS, _weighted_stats(vals, w)):
            out[f"comp_{stat}_{p}"] = v
    val = {o: np.array([el_tab["elements"][s][f"N{o}Valence"] or 0.0 for s in syms]) for o in "spdf"}
    avg = {o: float((val[o] * w).sum()) for o in "spdf"}
    tot = sum(avg.values()) or 1.0
    for o in "spdf":
        out[f"comp_avg_{o}_valence"] = avg[o]
        out[f"comp_frac_{o}_valence"] = avg[o] / tot
    out["comp_tm_fraction"] = float(sum(f for s, f in fracs.items() if basics()[s]["is_tm"]))
    return out


def shell_features(fracs: dict[str, float]) -> dict[str, float]:
    B = basics()
    syms = list(fracs)
    w = np.array([fracs[s] for s in syms])
    nd = np.array([B[s]["n_d"] for s in syms])
    nf = np.array([B[s]["n_f"] for s in syms])
    out = {}
    for tag, n, half, full in (("d", nd, 5, 10), ("f", nf, 7, 14)):
        mask = (n > 0) & (n < full)
        dev = n[mask] - half
        ww = w[mask] / w[mask].sum() if mask.any() else w[mask]
        out[f"shell_{tag}_dev_mean"] = float((dev * ww).sum()) if mask.any() else 0.0
        out[f"shell_{tag}_dev_abs_mean"] = float((np.abs(dev) * ww).sum()) if mask.any() else 0.0
        out[f"shell_{tag}_dev_sq_mean"] = float((dev ** 2 * ww).sum()) if mask.any() else 0.0
        out[f"shell_{tag}_dev_max"] = float(np.abs(dev).max()) if mask.any() else 0.0
        out[f"shell_is_{tag}_system"] = float(mask.any())
        out[f"shell_{tag}_atom_fraction"] = float(w[mask].sum())
    en = np.array([B[s]["X"] if B[s]["X"] is not None else 0.0 for s in syms], dtype=float)
    en_mean = float((en * w).sum())
    out["shell_en_contrast"] = float(en.max() - en.min())
    out["shell_en_std"] = float(np.sqrt(((en - en_mean) ** 2 * w).sum()))
    out["shell_n_elements"] = float(len(syms))
    z = np.array([B[s]["Z"] for s in syms], dtype=float)
    out["shell_z_max"], out["shell_z_min"] = float(z.max()), float(z.min())
    out["shell_mass_per_atom"] = float((np.array([B[s]["mass"] for s in syms]) * w).sum())
    return out


def symmetry_features(spacegroup: int) -> dict[str, float]:
    _, sg, _ = _tables()
    if str(int(spacegroup)) not in sg:
        raise ValueError(f"space group must be 1-230, got {spacegroup}")
    s = sg[str(int(spacegroup))]
    return {"sym_spacegroup": float(spacegroup), "sym_crystal_system": float(s["crystal_system"]),
            "sym_point_group_order": float(s["point_group_order"]),
            "sym_centrosymmetric": float(s["centrosymmetric"])}


def frac_features(fracs: dict[str, float]) -> dict[str, float]:
    """Atomic fraction of each element Z = 1..103 (frac_Z1 ... frac_Z103)."""
    out = {f"frac_Z{z}": 0.0 for z in range(1, 104)}
    for e, f in fracs.items():
        out[f"frac_Z{basics()[e]['Z']}"] = f
    return out


def mp_uses_u(fracs: dict[str, float]) -> bool:
    """Materials Project rule: GGA+U for oxides and fluorides of the listed metals."""
    _, _, mpu = _tables()
    return any(a in fracs for a in mpu["anions"]) and any(e in fracs for e in mpu["U"])


def settings_features(fracs: dict[str, float], pbe_plus_u: bool | None = None,
                      energy_criterion: bool = False) -> dict[str, float]:
    """Calculation settings that a standard MP PBE(+U) relaxation would use.

    pbe_plus_u=None applies the MP rule. energy_criterion flags the minority batch of MP
    calculations with a recorded energy-based ionic stopping criterion; new predictions
    use the standard (force-based) protocol, i.e. False.
    """
    _, _, mpu = _tables()
    use_u = mp_uses_u(fracs) if pbe_plus_u is None else bool(pbe_plus_u)
    us = {e: mpu["U"][e] for e in fracs if use_u and e in mpu["U"]}
    return {"set_pbe_plus_u": float(use_u), "set_u_max": max(us.values(), default=0.0),
            "set_u_total": float(sum(us.values())),
            "set_u_weighted": float(sum(fracs[e] * u for e, u in us.items())),
            "set_n_u_elements": float(len(us)), "set_energy_criterion": float(energy_criterion)}


def geometry_features(fracs: dict[str, float], volume_per_atom: float, cellpar=None,
                      prefix: str = "cell") -> dict[str, float]:
    """Scalars of a cell (A^3/atom; cellpar = a, b, c in A and alpha, beta, gamma in degrees).

    Definitions follow the paper's relaxed-geometry block; cellpar is taken in the setting
    the cell is given in, as it was for the training data.
    """
    B = basics()
    mass = sum(f * B[e]["mass"] for e, f in fracs.items())  # amu per atom
    radii = [B[e]["radius"] for e in fracs]
    pf = (4.0 * math.pi / 3.0 * sum(f * float(r) ** 3 for f, r in zip(fracs.values(), radii))
          / volume_per_atom) if all(r is not None for r in radii) else math.nan
    out = {f"{prefix}_volume_per_atom": volume_per_atom,
           # WHY: amu/A^3 -> g/cm^3 factor 1.66053906660
           f"{prefix}_density": mass / volume_per_atom * 1.66053906660,
           f"{prefix}_packing_fraction": pf}
    if cellpar is not None:
        a, b, c, al, be, ga = (float(v) for v in cellpar)
        out.update({f"{prefix}_alpha": al, f"{prefix}_beta": be, f"{prefix}_gamma": ga,
                    f"{prefix}_a_over_b": a / b, f"{prefix}_a_over_c": a / c,
                    f"{prefix}_b_over_c": b / c})
    return out


STRUCTURE_FORMATS = ("cif", "vasp", "extxyz", "xyz", "espresso-in", "aims", "res")


def read_structure(source: str, fmt: str | None = None):
    """ASE Atoms from a file path or from pasted file contents.

    fmt=None: from a path, ASE infers the format from the name (POSCAR, *.cif, *.xyz, ...);
    for pasted text, the formats in STRUCTURE_FORMATS are tried in turn.
    """
    import io
    import os

    from ase.io import read

    if len(source) < 4096 and "\n" not in source and os.path.exists(source):
        try:
            return read(source, format=fmt)
        except Exception:
            if fmt is not None:
                raise
            with open(source) as fh:
                text = fh.read()
    else:
        text = source
    errors = []
    for f in ((fmt,) if fmt else STRUCTURE_FORMATS):
        try:
            atoms = read(io.StringIO(text), format=f)
            if len(atoms) > 0 and atoms.cell.volume > 1e-6:
                return atoms
        except Exception as e:  # noqa: BLE001
            errors.append(f"{f}: {type(e).__name__}")
    raise ValueError("could not read the structure (tried " + ", ".join(errors) + "); "
                     "a periodic cell is required")


def spacegroup_of(atoms, symprec: float = 0.01) -> int:
    from ase.spacegroup.symmetrize import check_symmetry

    return int(check_symmetry(atoms, symprec=symprec).number)


def formula_of(atoms) -> str:
    return atoms.get_chemical_formula(mode="metal")


def finished_features(atoms, band_gap: float, magnetization: float, formation_energy: float | None = None,
                      e_above_hull: float | None = None) -> dict[str, float]:
    """Descriptors of a finished PBE(+U) relaxation (training columns fs_, fg_, fm_, fc_, fe_).

    atoms: the relaxed structure; band_gap in eV; magnetization: total magnetic moment of that cell in
    mu_B (its absolute value is used); formation_energy and e_above_hull in eV/atom (MP-compatible).
    """
    from volcorr.coordination import coordination_features

    if band_gap < 0:
        raise ValueError("band gap must be >= 0 eV")
    a, b, c, al, be, ga = (float(v) for v in atoms.cell.cellpar())
    vol = atoms.get_volume()
    mag_vol = abs(float(magnetization)) / vol  # mu_B / A^3
    x = {"fs_alpha": al, "fs_beta": be, "fs_gamma": ga, "fs_a_over_b": a / b, "fs_a_over_c": a / c,
         "fs_b_over_c": b / c, "fg_gap": float(band_gap), "fg_log1p_gap": math.log1p(float(band_gap)),
         "fg_gapped": float(band_gap > 0.1), "fm_mag_vol": mag_vol, "fm_mag_atom": mag_vol * vol / len(atoms),
         "fm_magnetic": float(mag_vol * vol / len(atoms) > 0.05)}
    x.update({"fc_" + k: v for k, v in coordination_features(atoms).items()})
    if formation_energy is not None and e_above_hull is not None:
        x.update({"fe_formation_energy": float(formation_energy), "fe_e_above_hull": float(e_above_hull)})
    return x


def build(formula: str, spacegroup: int | None = None, atoms=None, pbe_plus_u: bool | None = None,
          energy_criterion: bool = False) -> dict[str, float]:
    """All descriptors available from the given inputs (formula is always required).

    atoms (ase.Atoms, e.g. the experimental structure) adds the input-cell geometry and,
    when spacegroup is not given, the space group detected at symprec 0.01.
    """
    fracs = parse_formula(formula)
    x = {**composition_features(fracs), **shell_features(fracs),
         **settings_features(fracs, pbe_plus_u, energy_criterion)}
    if atoms is not None and spacegroup is None:
        spacegroup = spacegroup_of(atoms)
    if spacegroup is not None:
        x.update(symmetry_features(spacegroup))
    if atoms is not None:
        x.update(geometry_features(fracs, atoms.get_volume() / len(atoms), atoms.cell.cellpar()))
    return x
