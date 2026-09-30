# fsbench: exact feature subset references and a leak-free benchmark of binary metaheuristics

[![Code DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23070324.svg)](https://doi.org/10.5281/zenodo.23070324)
[![Data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23070390.svg)](https://doi.org/10.5281/zenodo.23070390)
[![Code licence: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Data licence: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)

Code, exact lookup tables and results for the accompanying manuscript.

## Contents
- `fsb/`: dataset loader, exhaustive KNN error tables, fitness oracle, the 13 methods, live evaluator
- `scripts/`: every step from raw data to figures, in the order listed in `reproduce.sh`
- `tests/`: 11 tests (tie rules, table correctness, planted optima, live evaluation versus exact tables)
- `PLAN.md`: analysis plan frozen before any metaheuristic run (git tag `plan-v1`)
- `results/`: run-level results (`runs.csv.gz`), summary statistics, `provenance.json`
- `reports/`: table verification (`verification.csv`) and byte-for-byte reproducibility (`reproducibility.csv`)
- `figures/`: all figures as vector PDF (fig2 to fig6 in the main text, figS1 to figS3 supplementary)

## Reproduce
Python 3.12 with the exact versions in `requirements.lock`:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.lock
    bash scripts/fetch_data.sh
    bash reproduce.sh

Approximate time on an Apple M2 MacBook Air: exact tables 15 min, table verification 15 min,
main runs 2 min, statistics and figures 3 min, larger-dataset replication 25 min.

## Data
The 18 benchmark files are fetched from github.com/ahmed-shameem/Feature_selection at commit
257d7dca4ae643733a69dcf06cb7c70b7cce37e2 and checked against the SHA-256 values in
`data/manifest.csv`. The source copy states no licence, so the files are fetched rather than
redistributed. Known issues are recorded, not altered: the Breastcancer file carries the sample
ID as its first column (dropped by the loader); BreastEW and CongressEW each have one row fewer
than the published table; duplicate rows per dataset are listed in the manifest.

## Rules fixed in advance
KNN with k = 5, Euclidean distance, min-max scaling fitted on the training part only, distances
quantised to 1e-9, distance ties to the lower training index, vote ties to the class of the
nearest tied neighbour (see the docstring of `fsb/tables.py`). Seeds are derived from the MD5 of
each run's identifiers, so every run is reproducible on its own.

## Provenance
`results/provenance.json` records the git commit, platform and library versions that produced
the results. `CHECKSUMS.txt` lists SHA-256 values of the main result files.


## Availability

Code, result files and the 18 data files of the suite: this repository, release
`paper-v1`, archived at https://doi.org/10.5281/zenodo.23070324 (MIT for code,
CC BY 4.0 for result files and tables).

The 120 exhaustive subset tables (91 MB uncompressed, 120 .npy arrays over
12 datasets and 5 outer folds, validation and test) are deposited separately at
https://doi.org/10.5281/zenodo.23070390 (CC BY 4.0). They are not stored in git
because they are large binary arrays; `reproduce.sh` rebuilds them in 984 s on
an Apple M2 MacBook Air.

Tags: `plan-v1` (16d510d, analysis plan frozen before any run), `eff18ed`
(methods, runner and analysis fixed before any run), `results-v1` (4e51919,
frozen results), `paper-v1` (archived release).

Data files in `data/raw` derive from the UCI Machine Learning Repository and are
credited in DATA-ATTRIBUTION.md.
