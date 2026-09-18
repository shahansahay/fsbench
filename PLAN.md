# Analysis plan v1 (frozen before any metaheuristic run)

Working title: Reference-anchored benchmarking of binary metaheuristic feature
selection against exhaustively computed optima

## Question
How far do binary metaheuristics fall from the exact optimum on the datasets
the field benchmarks on, how much does the common evaluation protocol inflate
reported accuracy, and how stable are the resulting rankings?

## Data
Canonical 18-dataset suite (EW versions), files from
github.com/ahmed-shameem/Feature_selection at commit
257d7dca4ae643733a69dcf06cb7c70b7cce37e2; SHA-256 in data/manifest.csv.
Exact reference (all 2^d - 1 subsets): Breastcancer, Tic-tac-toe, Exactly,
Exactly2, M-of-n, HeartEW, WineEW, CongressEW, Vote, Zoo, Lymphography, SpectEW.
Not run in this version: BreastEW, IonosphereEW, KrvskpEW, WaveformEW, SonarEW,
PenglungEW (stated as a limitation).
Recorded, not altered: the Breastcancer file carries the sample ID as its first
column (dropped); BreastEW has 568 rows and CongressEW 434 (published 569 and
435); duplicate rows per dataset as listed in the manifest.

## Protocol
Outer: 5 stratified folds (seeded round-robin per class). Inner: stratified
80/20 hold-out of the outer-train part.
KNN, k = 5, Euclidean, min-max scaling fitted on the training part only.
Per fold, two exhaustive tables of error counts over every subset:
E_val (inner-train to inner validation) and E_test (outer-train to outer test).
Tie rules: distances quantised to 1e-9; distance ties go to the lower training
index; vote ties go to the class of the nearest tied neighbour.
Protocol N (leak-free): fitness from E_val, reported accuracy from E_test.
Protocol L (common practice): fitness and reported accuracy both from E_test.
Fitness: f = 0.99 * err + 0.01 * |S| / d (sensitivity: 0.9 and 0.1).

## Methods
GA (native binary). PSO, GWO, WOA and SCA, each with an S-shaped and a V-shaped
transfer function, treated as separate methods.
Baselines: all features, random search, SFS, (1+1) EA (bit-flip rate 1/d,
accepts equal fitness).
Budget B0 = 1,000 fitness calls counted at the oracle, revisits included.
Budget curve: 100, 300, 1,000, 3,000.
30 seeds per method, dataset, fold and protocol; seed = first 8 bytes of
MD5(dataset|fold|protocol|method|base seed).
Hyperparameters from the originating papers, no tuning.

## Primary endpoints
P1 Hit rate (fitness equal to the exact optimum value) and percentile rank of
   the returned subset among all 2^d - 1 subsets.
P2 Leakage: reported accuracy under L minus test accuracy under N, in
   percentage points.
P3 Ranking agreement: Kendall tau between rankings by best-of-30, mean and
   median, and between validation fitness and leak-free test accuracy.

## Hypotheses (reported whichever way they fall)
H1 On d <= 13 datasets at B0, random search's hit rate is within 10 points of
   the best metaheuristic.
H2 No metaheuristic beats the (1+1) EA or SFS in leak-free test accuracy after
   Holm correction.
H3 Mean L-minus-N inflation exceeds the gap between the top two methods.
H4 Rankings by validation fitness and by leak-free test accuracy disagree
   (Kendall tau below 0.5).

## Statistics
Unit of inference: dataset (median over seeds within a fold, mean over folds).
Friedman omnibus; pairwise Wilcoxon signed-rank with Holm correction;
Vargha-Delaney A12 for run-level effects; cluster bootstrap over datasets
(10,000 resamples) for intervals. Anything not listed above is exploratory,
including the number-of-datasets subsampling analysis.
