# Trained models

The models were fitted to experimental volumes taken from ICSD structures (through their
Materials Project matches). The authors have confirmed that distributing the trained
models is permitted.

What is and is not distributed:
- Distributed: model parameters (tree ensembles), aggregate element and space-group counts
  (`domain.json`), and the bundled public lookup tables (Magpie elemental data, spglib
  space-group data, Materials Project U values).
- Not distributed: any ICSD entry, structure, volume, identifier or training row. The tool
  never reports nearest training neighbours, because a predicted error combined with the
  public Materials Project volume would reveal the ICSD volume of that entry.
