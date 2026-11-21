# Reproducibility and execution notes

## Status

This repository is an initial research snapshot. It is not yet a tagged archival
release and its outputs should not be treated as independently validated until
the checks below have been completed.

## Software environment

- Python: record the exact interpreter version used for the final run.
- Dependencies: see the root-level `requirements.txt`.
- For a reproducible archival release, freeze the tested package versions in a
  separate lock file (for example, `requirements-lock.txt`).

## Installation

From the repository root, create and activate a virtual environment, then run:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Research components

- `01_Algorithm/`: K-LCR and comparator implementations.
- `02_Simulation_Data/`: simulation generation and design-sensitivity scripts.
- `04_Experiments/`: merging and comparator experiments.
- `05_Results/`: stored outputs, tables, figures, and workbooks.

Run scripts from the repository root unless a script's own instructions state
otherwise. Before publication, add the exact command line and expected output
for every script used to generate a manuscript table or figure.

## Checks required before an archival release

1. Record exact Python and dependency versions.
2. Record all random seeds, replicate counts, and simulation parameters.
3. Document initial K, vertical initialization, tie-breaking, empty/small-cluster
   handling, convergence, maximum iterations, and cycle handling.
4. Verify the implementation of perpendicular-distance reassignment and OLS
   refitting against the final methodological description.
5. Verify that optional merging uses the absolute decrease in pooled R-squared;
   slope similarity is not a merging criterion.
6. Regenerate all manuscript tables and figures from a clean environment and
   compare them with the committed outputs.
7. Document each empirical dataset's original source, version, preprocessing,
   and redistribution conditions. Do not redistribute data without permission.
8. Keep the illustrative hypothetical dataset clearly distinguished from
   empirical and simulated study datasets.
9. Check that the repository contains no credentials, private data, or
   unpublished material not intended for public release.

## Scope

The present study concerns bivariate pattern recognition. Predictive performance
is not evaluated; prediction is reserved for future research.

## Data provenance note

The Fuel dataset is excluded from this initial upload pending source and
redistribution verification. Empirical dataset descriptions and preprocessing
must be reconciled with the final manuscript before a formal release.
