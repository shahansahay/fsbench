# Reference-anchored benchmarking of binary metaheuristic feature selection

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
