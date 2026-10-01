import math

import pytest

from volcorr import features as F


def test_mp_u_rule():
    assert F.mp_uses_u(F.parse_formula("Fe2O3"))
    assert F.mp_uses_u(F.parse_formula("NiF2"))
    assert not F.mp_uses_u(F.parse_formula("FeS2"))      # no O/F
    assert not F.mp_uses_u(F.parse_formula("ZnO"))       # Zn has no U
    s = F.settings_features(F.parse_formula("Fe2O3"))
    assert s["set_u_max"] == 5.3 and s["set_energy_criterion"] == 0.0


@pytest.mark.parametrize("sg,cs,order,centro", [(225, 7, 48, 1), (167, 5, 12, 1), (62, 3, 8, 1),
                                                (14, 2, 4, 1), (1, 1, 1, 0), (216, 7, 24, 0)])
def test_spacegroup_table(sg, cs, order, centro):
    s = F.symmetry_features(sg)
    assert (s["sym_crystal_system"], s["sym_point_group_order"], s["sym_centrosymmetric"]) == (cs, order, centro)


def test_shell_filling():
    x = F.shell_features(F.parse_formula("Fe2O3"))   # Fe 3d6 -> n_d - 5 = +1
    assert x["shell_d_dev_mean"] == 1.0 and x["shell_is_d_system"] == 1.0
    assert math.isclose(x["shell_d_atom_fraction"], 0.4)
    x = F.shell_features(F.parse_formula("ZnO"))     # full d shell does not count
    assert x["shell_is_d_system"] == 0.0


def test_formula_order_invariance():
    a, b = F.build("Fe2O3", 167), F.build("O3Fe2", 167)
    assert all(a[k] == b[k] or (a[k] != a[k] and b[k] != b[k]) for k in a)


def test_cell_geometry():
    fr = F.parse_formula("NaCl")
    g = F.geometry_features(fr, 22.4, [5.64, 5.64, 5.64, 90, 90, 90])
    assert math.isclose(g["cell_density"], 58.44 / 2 / 22.4 * 1.66054, rel_tol=1e-3)


POSCAR = """NaCl
1.0
0 2.82 2.82
2.82 0 2.82
2.82 2.82 0
Na Cl
1 1
Direct
0 0 0
0.5 0.5 0.5
"""


def test_read_pasted_poscar():
    atoms = F.read_structure(POSCAR)
    assert F.formula_of(atoms) == "NaCl" and F.spacegroup_of(atoms) == 225


def test_read_pasted_cif(tmp_path):
    from ase.io import write

    atoms = F.read_structure(POSCAR)
    p = tmp_path / "x.cif"
    write(p, atoms)
    assert F.spacegroup_of(F.read_structure(p.read_text())) == 225
    assert F.spacegroup_of(F.read_structure(str(p))) == 225


def test_frac_features():
    x = F.frac_features(F.parse_formula("Fe2O3"))
    assert len(x) == 103 and x["frac_Z26"] == 0.4 and x["frac_Z8"] == 0.6


@pytest.mark.parametrize("formula,expected", [
    ("Li0.5CoO2", {"Li": 0.5, "Co": 1, "O": 2}), ("CuSO4·5H2O", {"Cu": 1, "S": 1, "O": 9, "H": 10}),
    ("K3[Fe(CN)6]", {"K": 3, "Fe": 1, "C": 6, "N": 6}), ("Mg3(PO4)2", {"Mg": 3, "P": 2, "O": 8}),
    ("NaN", {"Na": 1, "N": 1})])
def test_formula_parser(formula, expected):
    assert F._parse_counts(formula) == pytest.approx(expected)
