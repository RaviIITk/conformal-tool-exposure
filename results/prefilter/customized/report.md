# Stage-1 prefilter benchmark (ToolRet)

982 steps (535 single-tool) over 3367 tools, catalogs: customized. Recall@K = all gold tools among the top K. `hybrid` = BM25 + dense fused with RRF (the current prefilter design). `expanded` docs = Tool-REX tool profiles. `query+instruction` uses ToolRet's target-aware instruction (written with the gold tools in view): an oracle upper bound only. ΔR@50 = paired bootstrap difference against the current prefilter (hybrid, MiniLM, ToolRet docs). Latency = CPU time to encode one query (4 cores, fp32, batch 16).

| method | encoder | docs | query | R@10 | R@20 | R@50 | R@100 | R@50 single | R@50 multi | ΔR@50 vs current [95% CI] | params (M) | query ms (CPU) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense | Tool-Embed-0.6B | toolret | query | 0.334 | 0.396 | 0.482 | 0.527 | 0.783 | 0.121 | +0.075 [+0.053, +0.098] | 595.777 | 322.316 |
| hybrid | Tool-Embed-0.6B | toolret | query | 0.317 | 0.372 | 0.459 | 0.519 | 0.744 | 0.119 | +0.053 [+0.034, +0.072] | 595.777 | 322.316 |
| hybrid | LFM2.5-Embedding-350M | toolret | query | 0.288 | 0.357 | 0.451 | 0.485 | 0.720 | 0.130 | +0.045 [+0.026, +0.063] | 354.484 | 154.524 |
| hybrid | ToolRet-trained-bge-base-en-v1.5 | toolret | query | 0.287 | 0.359 | 0.440 | 0.482 | 0.705 | 0.123 | +0.034 [+0.018, +0.049] | 109.482 | 47.799 |
| hybrid | Nemotron-3-Embed-1B-BF16 | toolret | query | 0.291 | 0.348 | 0.437 | 0.473 | 0.701 | 0.121 | +0.031 [+0.014, +0.048] | 1140.918 | 488.272 |
| dense | Nemotron-3-Embed-1B-BF16 | toolret | query | 0.316 | 0.373 | 0.435 | 0.490 | 0.697 | 0.121 | +0.029 [+0.010, +0.047] | 1140.918 | 488.272 |
| dense | ToolRet-trained-bge-base-en-v1.5 | toolret | query | 0.299 | 0.351 | 0.427 | 0.469 | 0.667 | 0.139 | +0.020 [-0.001, +0.040] | 109.482 | 47.799 |
| dense | LFM2.5-Embedding-350M | toolret | query | 0.294 | 0.354 | 0.421 | 0.479 | 0.675 | 0.116 | +0.014 [-0.007, +0.038] | 354.484 | 154.524 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | toolret | query | 0.264 | 0.338 | 0.406 | 0.457 | 0.656 | 0.107 | – | 22.713 | 9.392 |
| dense | multi-qa-MiniLM-L6-cos-v1 | toolret | query | 0.262 | 0.301 | 0.370 | 0.439 | 0.594 | 0.101 | -0.037 [-0.053, -0.020] | 22.713 | 9.392 |
| bm25 | – | toolret | query | 0.220 | 0.264 | 0.316 | 0.362 | 0.523 | 0.067 | -0.091 [-0.110, -0.071] | – | – |

## Recall@50 by catalog

| method | encoder | docs | query | customized |
|---|---|---|---|---|
| dense | Tool-Embed-0.6B | toolret | query | 0.482 |
| hybrid | Tool-Embed-0.6B | toolret | query | 0.459 |
| hybrid | LFM2.5-Embedding-350M | toolret | query | 0.451 |
| hybrid | ToolRet-trained-bge-base-en-v1.5 | toolret | query | 0.440 |
| hybrid | Nemotron-3-Embed-1B-BF16 | toolret | query | 0.437 |
| dense | Nemotron-3-Embed-1B-BF16 | toolret | query | 0.435 |
| dense | ToolRet-trained-bge-base-en-v1.5 | toolret | query | 0.427 |
| dense | LFM2.5-Embedding-350M | toolret | query | 0.421 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | toolret | query | 0.406 |
| dense | multi-qa-MiniLM-L6-cos-v1 | toolret | query | 0.370 |
| bm25 | – | toolret | query | 0.316 |
