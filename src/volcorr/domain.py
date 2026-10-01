"""Applicability-domain warnings from aggregate training statistics (no training entries exposed)."""
from __future__ import annotations

import json
import math
from functools import lru_cache
from importlib import resources


@lru_cache(maxsize=1)
def stats() -> dict:
    return json.loads((resources.files("volcorr") / "models" / "domain.json").read_text())


def check(fracs: dict[str, float], spacegroup: int | None, x: dict[str, float]) -> list[str]:
    s = stats()
    out = []
    for el in fracs:
        n = s["element_counts"].get(el, 0)
        if n == 0:
            out.append(f"{el} does not occur in the training data; the prediction is an extrapolation.")
        elif n < s["rare_element_threshold"]:
            out.append(f"{el} occurs in only {n} training compounds; expect a larger error.")
    if spacegroup is not None:
        n = s["spacegroup_counts"].get(str(spacegroup), 0)
        if n < s["rare_spacegroup_threshold"]:
            out.append(f"Space group {spacegroup} occurs in {n} training compounds; the symmetry "
                       "descriptors are poorly sampled there.")
    for k, (lo, hi) in s["feature_ranges"].items():
        v = x.get(k)
        if v is not None and not math.isnan(v) and not lo <= v <= hi:
            out.append(f"{k} = {v:.3g} lies outside the training range [{lo:.3g}, {hi:.3g}].")
    return out
