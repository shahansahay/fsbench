# Revision analysis plan (frozen before any revision analysis runs)

Companion to PLAN.md (tag plan-v1). The analyses below answer the reviews of the
manuscript and were written down, committed and tagged revision-plan-v1 before
any of them was run. All of them are exploratory and carry the labels X5 to X10
in the manuscript, continuing X1 to X4. Nothing in PLAN.md, in results/ as tagged
results-v1, or in the decisions on P1-P3 and H1-H4 changes. Every quantity below
is reported whichever way it falls. A further analysis can only be added by a
dated amendment to this file, committed before that analysis runs.

Common settings, unless a section says otherwise: the 12 exact datasets, 5 outer
folds, KNN k = 5 with the tie rules of fsb/tables.py, alpha = 0.99, 1,000 oracle
calls, the 12 search methods with 30 seeds and ALL once, dataset-level values
(median over seeds within a fold, mean over folds; mean over seeds for hit rate),
dataset as the unit of inference, 95% cluster-bootstrap intervals over datasets
(10,000 resamples, generator seed 20260919).

## X5 Matched control for inflation

Question: does the L-minus-N gap come from scoring the reported accuracy on rows
that also guided the search, or from the other differences between the two
protocols (classifier trained on all outer-training rows under L, on the
inner-training rows under N; a different number of scoring rows)?

Design: each outer test fold is split into two stratified halves A and B with
stratified_holdout(y_test, 0.5, seed_for(name, "half", fold, BASE_SEED)), A being
the rows the hold-out selects. Two exact tables per fold are built with the same
code and rules: e_A (outer-train to A) and e_B (outer-train to B). Scaling is fitted
on outer-train, as for e_test, so e_A(S) + e_B(S) = e_test(S) for every subset;
this identity is checked on all 120 pairs before anything else is computed.

Search runs: protocol "LA" takes fitness from e_A (n = |A|), protocol "LB" from
e_B. Every method runs with 30 seeds per dataset, fold and protocol, seeds
seed_for(dataset, fold, protocol, method, alpha, budget, rep, BASE_SEED).
For a subset S selected on half H, with H' the other half:
- leak component  = acc_H(S) - acc_H'(S)
- configuration component = acc_H'(S) - acc_H'(S_N), with S_N the subset the same
  method, fold and seed index returned under protocol N in the main runs
- their sum = acc_H(S) - acc_H'(S_N)
Each is averaged over the two directions (A to B, B to A), then reduced to
dataset level as above, for the 12-method mean, for each method, and for the
exact optimum of the half table (ties averaged).
Reported: means with intervals; the leak component as a multiple of the 0.69 pp
leak-free spread of the nine metaheuristic variants; a two-sided Wilcoxon
signed-rank test over datasets for each component (unadjusted, exploratory).
Decision rule for the wording of the manuscript: if the interval of the
configuration component for the 12-method mean lies within plus or minus 1 pp,
the inflation is described as attributable to the reuse of the scoring rows
under a matched control; otherwise the leak component is reported separately and
every ratio quoted for inflation is also given with the leak component.

## X6 Duplicate and conflicting rows

(a) Descriptive, per dataset and fold: the share of inner-validation rows with
an identical feature vector among the inner-training rows, and of outer-test rows
with one among the outer-training rows, split by same and different label.

(b) Sensitivity: on the 8 datasets with duplicates (Breastcancer, Exactly,
Exactly2, M-of-n, CongressEW, Vote, Zoo, SpectEW), keep the first occurrence of
every identical (features, label) row and drop every row whose feature vector
occurs with more than one label. Splits, tables and runs (protocols N and L) are
rebuilt for those datasets with the unchanged code and seed scheme, written to
separate folders. The other 4 datasets keep their original results. On this
deduplicated suite of 12 datasets: mean inflation with interval, the Friedman
test on leak-free accuracy, H1 to H4 by their frozen criteria, and the number of
datasets on which the 12-method mean falls below ALL leak-free (X3). The main
analysis remains the suite as distributed.

## X7 Power and equivalence for H2

For each of the 18 pairs of H2 (nine metaheuristic variants against the (1+1) EA
and against SFS), on dataset-level leak-free accuracy:
- minimum detectable difference: the 12 observed differences are centred on
  zero, a shift delta is added, 12 datasets are drawn with replacement, and the
  two-sided Wilcoxon test is applied; 2,000 draws per delta on a 0.1 pp grid,
  generator seed 20260921. Reported: the smallest delta with power of at least
  0.8 at alpha 0.05 and at alpha 0.05/18 (the first Holm step).
- equivalence: two one-sided exact Wilcoxon signed-rank tests of the differences
  against margins of plus and minus 1 pp (primary) and 0.5 pp (sensitivity);
  equivalence is declared when both one-sided P values are below 0.05. Also
  reported: the 90% cluster-bootstrap interval of the mean difference.
Equivalence is stated in the manuscript only for pairs that meet this rule.

## X8 Tie rules of scikit-learn

(a) For the 300 random subsets and the full set per table already compared in
verify_tables.py (same generator), the absolute difference in accuracy points
between the table and KNeighborsClassifier(n_neighbors=5, algorithm="brute"):
mean and maximum per dataset.
(b) For the 7 datasets with d <= 13 (Breastcancer, Tic-tac-toe, Exactly, Exactly2,
HeartEW, M-of-n, WineEW), complete validation and test tables are built with
scikit-learn's classifier instead of the fixed tie rules, on the same splits and
scaling. On these tables: the exact validation optimum's leak-free accuracy and
optimism, the inflation of the exact optimum, and protocol N and L runs of the 12
methods (30 seeds) with mean inflation and the Friedman test over the 7 datasets.
Reported next to the same quantities from the original tables on those 7 datasets.

## X9 Protocol shift against variation across datasets and methods

On dataset-level accuracy of the 12 search methods under protocols N and L
(12 x 12 x 2, from results-v1 only): sums of squares of the balanced three-way
layout for dataset, method, protocol and their interactions, as shares of the
total; the number of datasets on which mean inflation is positive, overall and per
method; per dataset, inflation divided by the leak-free range across the nine
metaheuristic variants; the median over datasets of the within-fold standard
deviation of leak-free accuracy across seeds, as a reference scale. Scaling:
Spearman correlation over the 12 datasets between mean inflation and
sqrt(ln(2^d - 1) / n_test), n_test being the mean outer-test size of the dataset.

## X10 Rebuild on a second platform

The original E_val and E_test tables of all datasets with d <= 16 (100 of the
120 tables) are rebuilt with the unchanged code under Linux on x86-64, and their
SHA-256 digests are compared with tables/index.csv, which records the tables built
on the Apple M2 machine. The X5 identity e_A + e_B = e_test, with e_A and e_B
built on the second machine, extends the comparison to every dataset run there.
Matches and mismatches are both reported; the statement on portability in the
manuscript follows the result.

## Outputs

Code: scripts/revision_*.py. Results: results/revision/ (tables as CSV, a summary
JSON, run files). New tables live outside tables/ so that results-v1 is untouched.
The results are tagged revision-results-v1 when complete.

## Amendment 1 (23 November 2026, committed before X8 runs on the Apple M2 machine)

X8(a) was run first on the Linux x86-64 machine. Replaying the generator of
verify_tables.py drew the same subsets (the counts of checked subsets match
reports/verification.csv), and the exact tables are identical on both machines
(their SHA-256 digests match tables/index.csv), but the share of subsets on which
scikit-learn 1.8.0 agreed with the tables did not match the recorded shares on
datasets with many distance ties (Tic-tac-toe, fold 0, validation table: 0.50 on
Linux against 0.29 recorded; Breastcancer, fold 0, test table: 0.76 against 0.72),
while it matched on HeartEW (0.9568 on both). For the same subsets, therefore,
scikit-learn's error counts differ between the two machines. Added analysis:

X8c. X8(a) is run on both machines, and the error counts scikit-learn returns for
the same subsets are compared between them: per dataset, the share of subsets with
identical counts and the number of tables with at least one difference. Results
are reported whichever way they fall. X8(a) and X8(b) are reported for the Apple
M2 machine, on which reports/verification.csv was produced, with the Linux values
given as a check. X5 and X6 depend only on the tables and on numpy's generators
and are run on both machines; their outputs are compared file by file.

## Amendment 2 (23 November 2026, committed before X11 runs)

Why: in X5 the configuration component of the 12-method mean was -1.79 pp with a
95% interval of -3.14 to -0.56 pp, so the decision rule of X5 was not met and the
leak component is reported separately, as that rule requires. That result stands
and is reported. Each half of a test fold holds about 10% of a dataset's rows,
fewer than the inner validation set (about 16%) on every fold, so the
configuration component of X5 combines the training rows of protocol L with a
smaller scoring set and cannot separate the two. X11 removes the difference in
size. It was specified after the results of X5 were seen, and it is reported
whichever way it falls. No other analysis is added.

## X11 Size-matched control for inflation

Design: each outer test fold T is dealt into five stratified parts P1 to P5 with
stratified_folds(y_test, 5, seed_for(name, "fifth", fold, BASE_SEED)). For each
part an exact table e_Pk (outer-train to Pk) is built with the code and rules of
fsb/tables.py. Scaling is fitted on outer-train, as for e_test, so the five part
tables sum to e_test for every subset; this identity is checked on all 60 folds
before anything else is computed. A search that leaves out Pk takes its fitness
from e_test - e_Pk, scored on n_test - |Pk| rows, about 16% of the dataset, the
size of the inner validation set. It therefore uses the classifier of protocol L
(trained on all outer-training rows) with a scoring set of the size protocol N
uses. The returned subset is scored on Pk, which took no part in the search.

Runs: for each part, the 12 methods with 30 seeds and ALL once, seeds
seed_for(dataset, fold, "X11P" + k, method, alpha, budget, rep, BASE_SEED);
1,000 oracle calls, alpha = 0.99; 12 x 5 x 5 x 361 = 108,300 runs.

Quantities, per method, fold and seed index (pooling the five parts):
- fresh accuracy: the test rows classified correctly by the subset chosen
  without them, summed over the five parts, divided by n_test;
- scoring accuracy: the mean over the five parts of the accuracy on the rows
  that scored the search;
- protocol N accuracy: the leak-free test accuracy of the subset returned under
  protocol N by the same method, fold and seed index (main runs);
- configuration component = fresh - protocol N; leak component = scoring -
  fresh; sum = scoring - protocol N.
Each is reduced as in the manuscript (median over seeds within a fold, mean over
folds), for the 12-method mean, per method and per dataset; and the same three
quantities for the exact optimum of each leave-one-part-out table (ties
averaged) against the exact validation optimum. Because medians are not
additive, the reduced sum can differ from the sum of the two reduced
components; the difference is reported. Reported: means with 95%
cluster-bootstrap intervals over datasets (10,000 resamples, generator seed
20260919), two-sided Wilcoxon signed-rank P over the 12 datasets (unadjusted,
exploratory), and the mean sizes of the scoring sets and of the inner
validation sets.

Decision rule for the wording of the manuscript: if the 95% interval of the
configuration component of the 12-method mean lies within plus or minus 1 pp,
the manuscript states that, with the scoring set matched in size, the training
configuration of protocol L changed leak-free accuracy by less than 1 pp and the
inflation is attributable to the reuse of the scoring rows. Otherwise the
configuration component is reported as the effect of the training configuration,
whatever its sign, and wherever the inflation is attributed it is described as
the sum of the leak and configuration components, with both values given.

Code: scripts/revision_x11.py (build, run, analyze), run on the Apple M2 machine.
Results: results/revision/x11_tables/index.csv (the part tables themselves are
not stored in git; their SHA-256 digests are in the index), x11_runs.csv.gz,
x11_*_pp.csv, x11_exact_optimum_*.csv, x11_scoring_sizes.csv and "X11" in
summary_revision.json, tagged revision-results-v2 when complete.
