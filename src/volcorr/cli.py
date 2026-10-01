"""Command line: volcorr Fe2O3 --sg 167 | volcorr --cif x.cif [--mace] | volcorr NaCl --sg 225 --vdft 22.9"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="volcorr", description=(
        "Predict how far a PBE/PBE+U (Materials Project settings) or MACE-MPA-0 relaxed volume "
        "deviates from experiment, and correct it."))
    p.add_argument("formula", nargs="?", help="chemical formula, e.g. Fe2O3")
    p.add_argument("--sg", type=int, help="space group number (1-230)")
    p.add_argument("--cif", help="experimental structure (CIF file)")
    p.add_argument("--vdft", type=float, help="your PBE(+U) volume, A^3/atom -> corrected volume")
    p.add_argument("--mace", action="store_true", help="relax the CIF with MACE-MPA-0 and correct it")
    p.add_argument("--device", default="cpu")
    u = p.add_mutually_exclusive_group()
    u.add_argument("--plus-u", dest="pbe_plus_u", action="store_true", default=None, help="force PBE+U")
    u.add_argument("--no-u", dest="pbe_plus_u", action="store_false", help="force plain PBE")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    a = p.parse_args(argv)
    if not a.formula and not a.cif:
        p.error("give a formula or --cif")

    from volcorr.predict import predict

    r = predict(a.formula, a.sg, a.cif, a.vdft, a.mace, a.pbe_plus_u, a.device)
    d = r.as_dict()
    if a.json:
        json.dump(d, sys.stdout, indent=1)
        print()
        return 0
    print(f"input:      {d['input']}")
    print(f"reference:  {d['reference']}")
    if d["settings_assumed"]:
        s = d["settings_assumed"]
        us = ", ".join(f"{k} {v}" for k, v in s["U_eV"].items())
        print(f"settings:   {s['functional']}{' (U: ' + us + ' eV)' if us else ''}")
    print(f"predicted volume error vs experiment: {r.delta:+.2f} %  "
          f"(68%: {r.interval68[0]:+.2f} to {r.interval68[1]:+.2f}; 90%: {r.interval90[0]:+.2f} to {r.interval90[1]:+.2f})")
    if r.v_corrected is not None:
        print(f"volume:     {r.v_ref:.3f} -> corrected {r.v_corrected:.3f} A^3/atom")
    if r.v_ref_predicted is not None:
        print(f"volume:     given cell {r.v_exp_input:.3f}; expected PBE(+U) volume "
              f"{r.v_ref_predicted:.3f} A^3/atom")
    for w in r.warnings:
        print(f"warning:    {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
