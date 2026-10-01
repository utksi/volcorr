import json
from importlib import resources

import pytest

from volcorr.predict import predict

REF = json.loads((resources.files("volcorr") / "models" / "reference_predictions.json").read_text())


@pytest.mark.parametrize("case", REF, ids=[c["id"] for c in REF])
def test_reference_predictions(case):
    r = predict(**case["kwargs"])
    assert r.tier == case["tier"]
    assert abs(r.delta - case["delta"]) < 1e-6


def test_interval_contains_point():
    r = predict("Fe2O3", 167)
    assert r.interval90[0] <= r.interval68[0] <= r.delta <= r.interval68[1] <= r.interval90[1]


def test_unknown_element_warns():
    from volcorr import domain

    w = domain.check({"Es": 0.5, "O": 0.5}, 225, {})
    assert any(m.startswith("Es ") for m in w)


def test_finished_routes():
    poscar = "NaCl\n1.0\n0 2.85 2.85\n2.85 0 2.85\n2.85 2.85 0\nNa Cl\n1 1\nDirect\n0 0 0\n0.5 0.5 0.5\n"
    f = predict(relaxed=poscar, band_gap=5.0, magnetization=0.0)
    g = predict(relaxed=poscar, band_gap=5.0, magnetization=0.0, formation_energy=-2.1, e_above_hull=0.0)
    assert f.tier == "F_finished" and g.tier == "G_finished_mp"
    assert abs(f.v_ref - 2 * 2.85 ** 3 / 2) < 1e-6 and f.v_corrected is not None
    with pytest.raises(ValueError):
        predict(relaxed=poscar, band_gap=5.0)


def test_coordination_rocksalt():
    from volcorr.coordination import coordination_features
    from volcorr.features import read_structure
    x = coordination_features(read_structure("NaCl\n1.0\n0 2.85 2.85\n2.85 0 2.85\n2.85 2.85 0\nNa Cl\n1 1\nDirect\n0 0 0\n0.5 0.5 0.5\n"))
    assert x["site_cn_mean"] == 6.0 and x["cn_hist_6"] == 1.0


def test_vdft_route():
    r = predict("NaCl", 225, v_dft=23.0)
    assert r.tier == "D_vdft" and r.v_corrected is not None
