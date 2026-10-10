# Prefilter K sweep: prefilter and final results by K

Steps: 12 dev + 47 eval (100 random 50/50 calibration/test splits per K). Prefilter: BM25 + `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` with RRF, top K kept. Score sources: dense, mercury-choice (API sources at K = 100).

## Prefilter results (recall = all gold tools among the K candidates)

| K | dev recall | eval recall (all) | eval recall (single-tool) | eval recall (multi-tool) |
|---|---|---|---|---|
| 10 | 0.167 | 0.426 | 0.426 | – |
| 20 | 0.333 | 0.489 | 0.489 | – |
| 30 | 0.333 | 0.553 | 0.553 | – |
| 50 | 0.500 | 0.596 | 0.596 | – |
| 100 | 0.667 | 0.681 | 0.681 | – |

## Final results (single-tool steps)

End to end (e2e) = all single-tool steps, prefilter misses count as failures; Stage 2 = steps whose gold tool survived the prefilter. hit@k = right tool among the top k tools shown; coverage and set size are for LAC conformal sets at α = 0.1.

| source | K | e2e hit@1 | e2e hit@3 | e2e hit@5 | Stage-2 n | Stage-2 hit@1 | Stage-2 hit@3 | Stage-2 hit@5 | Stage-2 coverage | Stage-2 avg set size | e2e coverage | e2e avg set size |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense | 10 | 0.170 | 0.255 | 0.319 | 20 | 0.400 | 0.600 | 0.750 | 0.902 | 8.682 | 0.427 | 10.000 |
| dense | 20 | 0.170 | 0.234 | 0.298 | 23 | 0.348 | 0.478 | 0.609 | 0.921 | 16.246 | 0.488 | 20.000 |
| dense | 30 | 0.170 | 0.234 | 0.255 | 26 | 0.308 | 0.423 | 0.462 | 0.921 | 26.505 | 0.552 | 30.000 |
| dense | 50 | 0.170 | 0.234 | 0.255 | 28 | 0.286 | 0.393 | 0.429 | 0.928 | 43.091 | 0.590 | 50.000 |
| dense | 100 | 0.170 | 0.234 | 0.255 | 32 | 0.250 | 0.344 | 0.375 | 0.941 | 83.675 | 0.673 | 100.000 |
| mercury-choice | 100 | 0.468 | 0.574 | 0.574 | 32 | 0.688 | 0.844 | 0.844 | 0.951 | 4.986 | 0.673 | 100.000 |

## Final results, all α (single-tool steps, LAC)

| source | K | view | α | n | coverage (mean) | coverage (5th pct) | avg set size | singleton | coverage capped k=5 | threshold finite |
|---|---|---|---|---|---|---|---|---|---|---|
| dense | 10 | stage2 | 0.050 | 20 | 1.000 | 1.000 | 10.000 | 0.000 | 0.743 | 0.000 |
| dense | 10 | stage2 | 0.100 | 20 | 0.902 | 0.700 | 8.682 | 0.000 | 0.743 | 1.000 |
| dense | 10 | stage2 | 0.200 | 20 | 0.810 | 0.495 | 7.094 | 0.006 | 0.729 | 1.000 |
| dense | 10 | e2e | 0.050 | 47 | 0.427 | 0.331 | 10.000 | 0.000 | 0.321 | 0.000 |
| dense | 10 | e2e | 0.100 | 47 | 0.427 | 0.331 | 10.000 | 0.000 | 0.321 | 0.000 |
| dense | 10 | e2e | 0.200 | 47 | 0.427 | 0.331 | 10.000 | 0.000 | 0.321 | 0.000 |
| dense | 20 | stage2 | 0.050 | 23 | 1.000 | 1.000 | 20.000 | 0.000 | 0.602 | 0.000 |
| dense | 20 | stage2 | 0.100 | 23 | 0.921 | 0.750 | 16.246 | 0.000 | 0.602 | 1.000 |
| dense | 20 | stage2 | 0.200 | 23 | 0.828 | 0.500 | 12.292 | 0.001 | 0.601 | 1.000 |
| dense | 20 | e2e | 0.050 | 47 | 0.488 | 0.375 | 20.000 | 0.000 | 0.299 | 0.000 |
| dense | 20 | e2e | 0.100 | 47 | 0.488 | 0.375 | 20.000 | 0.000 | 0.299 | 0.000 |
| dense | 20 | e2e | 0.200 | 47 | 0.488 | 0.375 | 20.000 | 0.000 | 0.299 | 0.000 |
| dense | 30 | stage2 | 0.050 | 26 | 1.000 | 1.000 | 30.000 | 0.000 | 0.449 | 0.000 |
| dense | 30 | stage2 | 0.100 | 26 | 0.921 | 0.688 | 26.505 | 0.000 | 0.449 | 1.000 |
| dense | 30 | stage2 | 0.200 | 26 | 0.837 | 0.538 | 22.810 | 0.000 | 0.449 | 1.000 |
| dense | 30 | e2e | 0.050 | 47 | 0.552 | 0.417 | 30.000 | 0.000 | 0.255 | 0.000 |
| dense | 30 | e2e | 0.100 | 47 | 0.552 | 0.417 | 30.000 | 0.000 | 0.255 | 0.000 |
| dense | 30 | e2e | 0.200 | 47 | 0.552 | 0.417 | 30.000 | 0.000 | 0.255 | 0.000 |
| dense | 50 | stage2 | 0.050 | 28 | 1.000 | 1.000 | 50.000 | 0.000 | 0.420 | 0.000 |
| dense | 50 | stage2 | 0.100 | 28 | 0.928 | 0.714 | 43.091 | 0.000 | 0.420 | 1.000 |
| dense | 50 | stage2 | 0.200 | 28 | 0.789 | 0.568 | 31.085 | 0.000 | 0.420 | 1.000 |
| dense | 50 | e2e | 0.050 | 47 | 0.590 | 0.458 | 50.000 | 0.000 | 0.255 | 0.000 |
| dense | 50 | e2e | 0.100 | 47 | 0.590 | 0.458 | 50.000 | 0.000 | 0.255 | 0.000 |
| dense | 50 | e2e | 0.200 | 47 | 0.590 | 0.458 | 50.000 | 0.000 | 0.255 | 0.000 |
| dense | 100 | stage2 | 0.050 | 32 | 1.000 | 1.000 | 100.000 | 0.000 | 0.381 | 0.000 |
| dense | 100 | stage2 | 0.100 | 32 | 0.941 | 0.812 | 83.675 | 0.000 | 0.381 | 1.000 |
| dense | 100 | stage2 | 0.200 | 32 | 0.812 | 0.622 | 64.366 | 0.000 | 0.381 | 1.000 |
| dense | 100 | e2e | 0.050 | 47 | 0.673 | 0.542 | 100.000 | 0.000 | 0.255 | 0.000 |
| dense | 100 | e2e | 0.100 | 47 | 0.673 | 0.542 | 100.000 | 0.000 | 0.255 | 0.000 |
| dense | 100 | e2e | 0.200 | 47 | 0.673 | 0.542 | 99.882 | 0.000 | 0.255 | 0.010 |
| mercury-choice | 100 | stage2 | 0.050 | 32 | 1.000 | 1.000 | 100.000 | 0.000 | 0.849 | 0.000 |
| mercury-choice | 100 | stage2 | 0.100 | 32 | 0.951 | 0.809 | 4.986 | 0.552 | 0.848 | 1.000 |
| mercury-choice | 100 | stage2 | 0.200 | 32 | 0.828 | 0.562 | 2.203 | 0.616 | 0.801 | 1.000 |
| mercury-choice | 100 | e2e | 0.050 | 47 | 0.673 | 0.542 | 100.000 | 0.000 | 0.573 | 0.000 |
| mercury-choice | 100 | e2e | 0.100 | 47 | 0.673 | 0.542 | 100.000 | 0.000 | 0.573 | 0.000 |
| mercury-choice | 100 | e2e | 0.200 | 47 | 0.673 | 0.542 | 99.048 | 0.004 | 0.573 | 0.010 |

## API usage

| source | K | calls | input tokens | median latency (s) | p95 latency (s) | refusals |
|---|---|---|---|---|---|---|
| mercury-choice | 100 | 47 | 251442 | 0.921 | 1.402 | 0 |

Figures: `recall_vs_k.png`, `coverage_vs_k.png`, `set_size_vs_k.png`.
