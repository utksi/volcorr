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


def test_vdft_route():
    r = predict("NaCl", 225, v_dft=23.0)
    assert r.tier == "D_vdft" and r.v_corrected is not None
