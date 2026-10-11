# End to end: deployment prefilter + Mercury Decide

50 single-tool steps from ToolRet code, customized (stratified sample), one Mercury Decide call each. Prefilter: the step's own toolset, ranked by an RRF ensemble of `Lux1997/Tool-Embed-0.6B+mangopy/ToolRet-trained-bge-base-en-v1.5` over expanded tool docs. K = 200, the smallest K whose recall on the other 1886 steps reaches 0.95 + 0.02.

## Prefilter

| K | recall on the other steps |
|---|---|
| 50 | 0.899 |
| 100 | 0.947 |
| 150 | 0.964 |
| 200 | 0.971 |
| 250 | 0.978 |
| 300 | 0.986 |

On the Mercury sample at K = 200: recall 0.980 (exact tool id), 0.980 (a same-description copy also counts).

## Mercury Decide: right tool among its top k (end to end)

| match | hit@1 | hit@3 | hit@5 | hit@10 | hit@1 | prefilter hit | hit@5 | prefilter hit |
|---|---|---|---|---|---|---|
| exact | 0.460 | 0.580 | 0.600 | 0.720 | 0.469 | 0.612 |
| duplicate | 0.460 | 0.580 | 0.600 | 0.720 | 0.469 | 0.612 |

ECE of the top choice: 0.282.

## Conformal tool sets (LAC), end to end

Leave-one-out: each step's threshold is calibrated on the other n − 1 steps. `split` = 200 random 50/50 calibration/test splits, for comparison (needs n/2 calibration points, so small α gives no finite threshold). Theory 5th pct = Beta-distribution lower band for n − 1 calibration points.

| match | α | target | coverage (LOO) | theory 5th pct | avg set size | median size | singleton | empty | coverage (split) | split threshold finite |
|---|---|---|---|---|---|---|---|---|---|---|
| exact | 0.020 | 0.980 | 0.980 | 0.941 | 200.000 | 200.000 | 0.000 | 0.000 | 0.978 | 0.000 |
| exact | 0.050 | 0.950 | 0.960 | 0.907 | 177.300 | 200.000 | 0.000 | 0.000 | 0.959 | 0.540 |
| exact | 0.100 | 0.900 | 0.900 | 0.823 | 63.360 | 28.500 | 0.040 | 0.000 | 0.921 | 1.000 |
| exact | 0.200 | 0.800 | 0.800 | 0.701 | 30.460 | 6.500 | 0.240 | 0.000 | 0.810 | 1.000 |
| duplicate | 0.020 | 0.980 | 0.980 | 0.941 | 200.000 | 200.000 | 0.000 | 0.000 | 0.978 | 0.000 |
| duplicate | 0.050 | 0.950 | 0.960 | 0.907 | 177.300 | 200.000 | 0.000 | 0.000 | 0.959 | 0.540 |
| duplicate | 0.100 | 0.900 | 0.900 | 0.823 | 63.360 | 28.500 | 0.040 | 0.000 | 0.921 | 1.000 |
| duplicate | 0.200 | 0.800 | 0.800 | 0.701 | 30.460 | 6.500 | 0.240 | 0.000 | 0.810 | 1.000 |

## Stage-2 alternatives on the same candidates (exact match, leave-one-out)

`retrieval` = softmax over the candidates' mean cosine, temperature fit on the other steps.

| Stage-2 score | hit@1 | hit@5 | hit@10 | hit@20 | coverage α=0.05 | size α=0.05 | coverage α=0.1 | size α=0.1 | coverage α=0.2 | size α=0.2 |
|---|---|---|---|---|---|---|---|---|---|---|
| mercury | 0.460 | 0.600 | 0.720 | 0.800 | 0.960 | 177.300 | 0.900 | 63.360 | 0.800 | 30.460 |
| retrieval | 0.380 | 0.480 | 0.580 | 0.640 | 0.960 | 185.880 | 0.900 | 85.180 | 0.800 | 49.540 |
| 0.5 mercury + 0.5 retrieval | 0.460 | 0.560 | 0.640 | 0.700 | 0.960 | 189.440 | 0.900 | 73.040 | 0.800 | 43.480 |

## API usage

| calls | input tokens | median latency (s) | p95 latency (s) | refusals |
|---|---|---|---|---|
| 50 | 796925 | 1.783 | 3.327 | 0 |
