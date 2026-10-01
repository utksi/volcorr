"""Record predictions of the shipped models for a few textbook compounds (regression tests).

Run after train_final.py. Inputs are public (formula, space group, a round-number volume);
nothing here comes from the training data.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from volcorr.predict import predict  # noqa: E402

CASES = [("NaCl_A", {"formula": "NaCl"}), ("NaCl_B", {"formula": "NaCl", "spacegroup": 225}),
         ("Fe2O3_B", {"formula": "Fe2O3", "spacegroup": 167}), ("Si_B", {"formula": "Si", "spacegroup": 227}),
         ("MnO_B_noU", {"formula": "MnO", "spacegroup": 225, "pbe_plus_u": False}),
         ("NaCl_D", {"formula": "NaCl", "spacegroup": 225, "v_dft": 23.0}),
         ("NaCl_F", {"relaxed": "NaCl\n1.0\n0 2.85 2.85\n2.85 0 2.85\n2.85 2.85 0\nNa Cl\n1 1\nDirect\n0 0 0\n0.5 0.5 0.5\n",
                     "band_gap": 5.0, "magnetization": 0.0}),
         ("NaCl_G", {"relaxed": "NaCl\n1.0\n0 2.85 2.85\n2.85 0 2.85\n2.85 2.85 0\nNa Cl\n1 1\nDirect\n0 0 0\n0.5 0.5 0.5\n",
                     "band_gap": 5.0, "magnetization": 0.0, "formation_energy": -2.1, "e_above_hull": 0.0})]
out = []
for cid, kw in CASES:
    r = predict(**kw)
    out.append({"id": cid, "kwargs": kw, "tier": r.tier, "delta": r.delta})
    print(f"{cid:10s} {r.tier:13s} delta {r.delta:+.3f} %  warnings {len(r.warnings)}")
(ROOT / "src/volcorr/models/reference_predictions.json").write_text(json.dumps(out, indent=1))
