# End to end: deployment prefilter + Mercury Decide

50 single-tool steps from ToolRet code, customized (stratified sample), one Mercury Decide call each. Prefilter: the step's own toolset, ranked by `base + ToolRet-trained-e5-large-v2 + Gorilla train requests (centroid, weight chosen on other folds) + usage kNN (gated)` (scripts/run_prefilter_k50.py). K = 50 (set by --k).

## Prefilter

| K | recall on the other steps |
|---|---|
| 50 | 0.952 |
| 100 | 0.974 |
| 150 | 0.983 |
| 200 | 0.989 |

On the Mercury sample at K = 50: recall 0.940 (exact tool id), 0.940 (a same-description copy also counts).

## Mercury Decide: right tool among its top k (end to end)

| match | hit@1 | hit@3 | hit@5 | hit@10 | hit@1 | prefilter hit | hit@5 | prefilter hit |
|---|---|---|---|---|---|---|
| exact | 0.440 | 0.600 | 0.660 | 0.780 | 0.468 | 0.702 |
| duplicate | 0.440 | 0.600 | 0.660 | 0.780 | 0.468 | 0.702 |

ECE of the top choice: 0.283.

## Conformal tool sets (LAC), end to end

Leave-one-out: each step's threshold is calibrated on the other n − 1 steps. `split` = 200 random 50/50 calibration/test splits, for comparison (needs n/2 calibration points, so small α gives no finite threshold). Theory 5th pct = Beta-distribution lower band for n − 1 calibration points.

| match | α | target | coverage (LOO) | theory 5th pct | avg set size | median size | singleton | empty | coverage (split) | split threshold finite |
|---|---|---|---|---|---|---|---|---|---|---|
| exact | 0.020 | 0.980 | 0.940 | 0.941 | 50.000 | 50.000 | 0.000 | 0.000 | 0.937 | 0.000 |
| exact | 0.050 | 0.950 | 0.940 | 0.907 | 50.000 | 50.000 | 0.000 | 0.000 | 0.934 | 0.130 |
| exact | 0.100 | 0.900 | 0.900 | 0.823 | 35.080 | 39.500 | 0.020 | 0.000 | 0.916 | 0.505 |
| exact | 0.200 | 0.800 | 0.800 | 0.701 | 9.320 | 3.000 | 0.340 | 0.000 | 0.799 | 1.000 |
| duplicate | 0.020 | 0.980 | 0.940 | 0.941 | 50.000 | 50.000 | 0.000 | 0.000 | 0.937 | 0.000 |
| duplicate | 0.050 | 0.950 | 0.940 | 0.907 | 50.000 | 50.000 | 0.000 | 0.000 | 0.934 | 0.130 |
| duplicate | 0.100 | 0.900 | 0.900 | 0.823 | 35.080 | 39.500 | 0.020 | 0.000 | 0.916 | 0.505 |
| duplicate | 0.200 | 0.800 | 0.800 | 0.701 | 9.320 | 3.000 | 0.340 | 0.000 | 0.799 | 1.000 |

## Stage-2 alternatives on the same candidates (exact match, leave-one-out)

`retrieval` = softmax over the candidates' prefilter scores (the saved ranking's fused score), temperature fit on the other steps.

| Stage-2 score | hit@1 | hit@5 | hit@10 | hit@20 | coverage α=0.05 | size α=0.05 | coverage α=0.1 | size α=0.1 | coverage α=0.2 | size α=0.2 |
|---|---|---|---|---|---|---|---|---|---|---|
| mercury | 0.440 | 0.660 | 0.780 | 0.860 | 0.940 | 50.000 | 0.900 | 35.080 | 0.800 | 9.320 |
| retrieval | 0.440 | 0.620 | 0.780 | 0.900 | 0.940 | 50.000 | 0.900 | 39.320 | 0.800 | 20.260 |
| 0.5 mercury + 0.5 retrieval | 0.460 | 0.660 | 0.760 | 0.880 | 0.940 | 50.000 | 0.900 | 39.760 | 0.800 | 12.340 |

## API usage

| calls | input tokens | median latency (s) | p95 latency (s) | refusals |
|---|---|---|---|---|
| 50 | 206352 | 1.017 | 2.111 | 0 |
