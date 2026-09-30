# Attribution for the data files in data/raw

The 18 comma-separated files in `data/raw` are not original to this study and
are redistributed here so that the analysis can be reproduced exactly.

## Immediate source

The files were obtained from the public repository
<https://github.com/ahmed-shameem/Feature_selection> at commit
`257d7dca4ae643733a69dcf06cb7c70b7cce37e2`. Each file was checked against the
SHA-256 digest recorded in `data/manifest.csv`. The files are byte-for-byte as
obtained, with one exception noted in the manuscript: the loader drops the
first column of `BreastCancer.csv`, which is a sample identifier, and the file
itself is unaltered.

## Original source

The datasets derive from the UCI Machine Learning Repository,
<https://archive.ics.uci.edu>, where they are published under the Creative
Commons Attribution 4.0 International Licence (CC BY 4.0). That licence
permits sharing and adaptation for any purpose provided appropriate credit is
given. Credit is given here to the UCI Machine Learning Repository and to the
donors and creators of each dataset.

Files, with the dataset each corresponds to:

| File | Dataset |
|---|---|
| BreastCancer.csv | Breast Cancer Wisconsin (Original) |
| BreastEW.csv | Breast Cancer Wisconsin (Diagnostic) |
| CongressEW.csv | Congressional Voting Records |
| Exactly.csv | Exactly (synthetic concept) |
| Exactly2.csv | Exactly2 (synthetic concept) |
| HeartEW.csv | Heart Disease |
| Ionosphere.csv | Ionosphere |
| KrVsKpEW.csv | Chess (King-Rook vs. King-Pawn) |
| Lymphography.csv | Lymphography |
| M-of-n.csv | M-of-N (synthetic concept) |
| PenglungEW.csv | Lung Cancer (Pen) |
| Sonar.csv | Connectionist Bench (Sonar, Mines vs. Rocks) |
| SpectEW.csv | SPECT Heart |
| Tic-tac-toe.csv | Tic-Tac-Toe Endgame |
| Vote.csv | Congressional Voting Records (subset) |
| WaveformEW.csv | Waveform Database Generator |
| Wine.csv | Wine |
| Zoo.csv | Zoo |

The `EW` suffix and the row and column counts follow the immediate source, not
the UCI originals. Two counts differ from the published tables and are
reported as distributed: `BreastEW.csv` has 568 rows against a published 569,
and `CongressEW.csv` has 434 against a published 435.

## Additional acknowledgement required by the source

The Lymphography dataset was obtained from the University Medical Centre,
Institute of Oncology, Ljubljana. Thanks go to M. Zwitter and M. Soklic for
providing the data.

## If you prefer not to use the copies here

`scripts/fetch_data.sh` downloads the files from the immediate source at the
pinned commit and verifies them against `data/manifest.csv`. The analysis runs
identically either way.
