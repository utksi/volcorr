"""Generate the lookup tables bundled with volcorr (run once; outputs are committed).

- elements.json:    22 Magpie elemental properties per element (matminer data, BSD licence)
- spacegroups.json: crystal system, point-group order and centrosymmetry for 1-230 (spglib)
- mp_u.json:        Materials Project GGA+U values and the rule that triggers them
"""
import json
import math
from pathlib import Path

import spglib
from matminer.featurizers.composition import ElementProperty
from matminer.utils.data import MagpieData
from pymatgen.core.periodic_table import Element

OUT = Path(__file__).resolve().parents[1] / "src" / "volcorr" / "data"

props = list(ElementProperty.from_preset("magpie").features)
magpie = MagpieData(impute_nan=False)
elements = {}
for z in range(1, 104):
    sym = Element.from_Z(z).symbol
    row = {}
    for p in props:
        try:
            v = float(magpie.get_elemental_property(Element(sym), p))
        except Exception:
            v = float("nan")
        row[p] = None if math.isnan(v) else v
    elements[sym] = row
(OUT / "elements.json").write_text(json.dumps({"properties": props, "elements": elements}, indent=0))

def crystal_system(n):
    for hi, name in ((2, 1), (15, 2), (74, 3), (142, 4), (167, 5), (194, 6), (230, 7)):
        if n <= hi:
            return name
sg = {}
for hall in range(1, 531):
    t = spglib.get_spacegroup_type(hall)
    n = t.number
    if str(n) in sg:
        continue
    sym = spglib.get_symmetry_from_database(hall)
    rots = {tuple(r.flatten()) for r in sym["rotations"]}
    inv = tuple((-1 * __import__("numpy").eye(3, dtype=int)).flatten())
    sg[str(n)] = {"symbol": t.international_short, "crystal_system": crystal_system(n),
                  "point_group": t.pointgroup_international, "point_group_order": len(rots),
                  "centrosymmetric": inv in rots}
assert len(sg) == 230
(OUT / "spacegroups.json").write_text(json.dumps(sg, indent=0))

mp_u = {"rule": "GGA+U if the composition contains O or F and any element listed here "
                "(Materials Project MPRelaxSet); U on the d shell",
        "anions": ["O", "F"],
        "U": {"Co": 3.32, "Cr": 3.7, "Fe": 5.3, "Mn": 3.9, "Mo": 4.38, "Ni": 6.2, "V": 3.25, "W": 6.2}}
(OUT / "mp_u.json").write_text(json.dumps(mp_u, indent=1))
print("wrote", sorted(p.name for p in OUT.iterdir()))

# Per-element basics so the featurizer runs without pymatgen (e.g. in the browser):
# Z, Pauling electronegativity (None if undefined), atomic mass (amu), empirical atomic
# radius (A, None if undefined), transition-metal flag, valence d and f counts in the
# convention of the shell-filling descriptors.


def _outer_shell_counts(sym: str) -> tuple[int, int]:
    """(n_d, n_f): highest-n s/p shell, d from n_max or n_max-1, f from n_max-1 or n_max-2."""
    try:
        es = Element(sym).full_electronic_structure
    except Exception:
        return 0, 0
    by_n: dict = {}
    for n, l, e in es:
        by_n.setdefault(n, {}).setdefault(l, 0)
        by_n[n][l] += e
    nmax = max(by_n)
    n_d = next((by_n[n]["d"] for n in (nmax, nmax - 1) if n in by_n and "d" in by_n[n]), 0)
    n_f = next((by_n[n]["f"] for n in (nmax - 1, nmax - 2) if n in by_n and "f" in by_n[n]), 0)
    return n_d, n_f


basics = {}
for z in range(1, 104):
    el = Element.from_Z(z)
    x = el.X
    r = el.atomic_radius
    nd, nf = _outer_shell_counts(el.symbol)
    basics[el.symbol] = {"Z": z, "X": None if x != x else float(x), "mass": float(el.atomic_mass),
                         "radius": None if r is None else float(r), "is_tm": bool(el.is_transition_metal),
                         "n_d": int(nd), "n_f": int(nf)}
(OUT / "element_basics.json").write_text(json.dumps(basics, indent=0))
print("wrote element_basics.json")
