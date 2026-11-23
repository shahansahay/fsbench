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
The 18 benchmark files come from github.com/ahmed-shameem/Feature_selection at commit
257d7dca4ae643733a69dcf06cb7c70b7cce37e2, are included in `data/raw` (they derive from the UCI
Machine Learning Repository, CC BY 4.0; see DATA-ATTRIBUTION.md), and are checked against the
SHA-256 values in `data/manifest.csv`; `scripts/fetch_data.sh` downloads them again at that commit
and repeats the check. Known issues are recorded, not altered: the Breastcancer file carries the sample
ID as its first column (dropped by the loader); BreastEW and CongressEW each have one row fewer
than the published table; duplicate rows per dataset are listed in the manifest.

## Rules fixed in advance
KNN with k = 5, Euclidean distance, min-max scaling fitted on the training part only, distances
quantised to 1e-9, distance ties to the lower training index, vote ties to the class of the
nearest tied neighbour (see the docstring of `fsb/tables.py`). Seeds are derived from the MD5 of
each run's identifiers, so every run is reproducible on its own.

## Analyses added after the results were frozen (X5 to X11)
`REVISION_PLAN.md` was committed and tagged `revision-plan-v1` (e856581, 23 November 2026) before any
of these analyses ran. Two amendments followed, each committed and tagged before the analysis it adds
was run: `revision-plan-amendment-1` added X8c before the runs on the Apple M2 machine, and
`revision-plan-amendment-2` (b676f81) added the size-matched control X11 after the result of X5 was
known. All of these analyses are exploratory. Their results are in `results/revision/`
(`summary_revision.json` collects every number); X5 to X10 are tagged `revision-results-v1`, and X11
and the Linux run of X8b are tagged `revision-results-v2`.

    python scripts/revision_x7_x9.py                          # X7 power and equivalence, X9 variance
    python scripts/revision_x5.py build && python scripts/revision_x5.py run --workers 8 && python scripts/revision_x5.py analyze
    python scripts/revision_x6.py describe && python scripts/revision_x6.py build && python scripts/revision_x6.py run --workers 8 && python scripts/revision_x6.py analyze
    python scripts/revision_x8.py compare --workers 8 && python scripts/revision_x8.py build --workers 8 && python scripts/revision_x8.py run --workers 8 && python scripts/revision_x8.py analyze
    python scripts/revision_x8.py compare_platforms           # X8c, needs the subset files of both machines
    python scripts/revision_x10.py                            # X10, rebuild of the tables with d <= 16
    python scripts/revision_x11.py build && python scripts/revision_x11.py run --workers 8 && python scripts/revision_x11.py analyze

The half tables (X5), the deduplicated tables (X6) and the fifth-part tables (X11) are rebuilt by
these scripts and are not stored in git; their SHA-256 values are in
`results/revision/x5_tables/index.csv`, `results/revision/x6_tables/index.csv` and
`results/revision/x11_tables/index.csv`. Files whose names end in `darwin-arm64` come from the Apple M2
machine and files ending in `linux-x86_64` from a Linux virtual machine; X8 writes one set per machine
because scikit-learn's error counts differ between the two (X8c).

## Provenance
`results/provenance.json` records the git commit, platform and library versions that produced
the results. `CHECKSUMS.txt` lists SHA-256 values of the main result files.


## Availability

Code, result files and the 18 data files of the suite: this repository, releases
`paper-v1`, `paper-v2` and `paper-v3`, archived under the concept DOI https://doi.org/10.5281/zenodo.23070324 (MIT for code,
CC BY 4.0 for result files and tables).

The 120 exhaustive subset tables (91 MB uncompressed, 120 .npy arrays over
12 datasets and 5 outer folds, validation and test) are deposited separately at
https://doi.org/10.5281/zenodo.23070390 (CC BY 4.0). They are not stored in git
because they are large binary arrays; `reproduce.sh` rebuilds them in 984 s on
an Apple M2 MacBook Air.

Tags: `plan-v1` (16d510d, analysis plan frozen before any run), `eff18ed`
(methods, runner and analysis fixed before any run), `results-v1` (4e51919,
frozen results), `paper-v1` (archived release), `revision-plan-v1` (e856581,
plan of X5 to X10 frozen before they ran), `revision-plan-amendment-1`,
`revision-results-v1`, `paper-v2` (archived release), `revision-plan-amendment-2`
(b676f81, X11 specified before it ran), `revision-results-v2`, and `paper-v3`
(archived release that matches the manuscript).

Data files in `data/raw` derive from the UCI Machine Learning Repository and are
credited in DATA-ATTRIBUTION.md.
