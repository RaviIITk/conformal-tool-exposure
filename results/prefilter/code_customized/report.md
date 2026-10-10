# Stage-1 prefilter benchmark (ToolRet)

2731 steps (1936 single-tool) over 7161 tools, catalogs: code, customized. Recall@K = all gold tools among the top K. `hybrid` = BM25 + dense fused with RRF (the current prefilter design). `expanded` docs = Tool-REX tool profiles. `query+instruction` uses ToolRet's target-aware instruction (written with the gold tools in view): an oracle upper bound only. ΔR@50 = paired bootstrap difference against the current prefilter (hybrid, MiniLM, ToolRet docs). Latency = CPU time to encode one query (4 cores, fp32, batch 16).

| method | encoder | docs | query | R@10 | R@20 | R@50 | R@100 | R@50 single | R@50 multi | ΔR@50 vs current [95% CI] | params (M) | query ms (CPU) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense | Tool-Embed-0.6B | expanded | query | 0.399 | 0.486 | 0.593 | 0.651 | 0.699 | 0.335 | +0.089 [+0.072, +0.106] | 595.777 | 354.705 |
| dense | Tool-Embed-0.6B | toolret | query | 0.415 | 0.484 | 0.585 | 0.637 | 0.670 | 0.380 | +0.081 [+0.064, +0.098] | 595.777 | 354.705 |
| hybrid | Tool-Embed-0.6B | expanded | query | 0.409 | 0.476 | 0.585 | 0.654 | 0.693 | 0.325 | +0.081 [+0.067, +0.096] | 595.777 | 354.705 |
| hybrid | Tool-Embed-0.6B | toolret | query | 0.399 | 0.490 | 0.584 | 0.650 | 0.676 | 0.361 | +0.080 [+0.067, +0.094] | 595.777 | 354.705 |
| dense | ToolRet-trained-bge-base-en-v1.5 | expanded | query | 0.389 | 0.470 | 0.560 | 0.621 | 0.689 | 0.248 | +0.056 [+0.042, +0.070] | 109.482 | 56.765 |
| hybrid | ToolRet-trained-bge-base-en-v1.5 | expanded | query | 0.391 | 0.461 | 0.558 | 0.618 | 0.690 | 0.238 | +0.054 [+0.041, +0.066] | 109.482 | 56.765 |
| hybrid | ToolRet-trained-bge-base-en-v1.5 | toolret | query | 0.384 | 0.460 | 0.544 | 0.592 | 0.673 | 0.231 | +0.040 [+0.029, +0.051] | 109.482 | 56.765 |
| hybrid | LFM2.5-Embedding-350M | expanded | query | 0.378 | 0.443 | 0.530 | 0.590 | 0.653 | 0.229 | +0.026 [+0.013, +0.039] | 354.484 | 158.015 |
| dense | ToolRet-trained-bge-base-en-v1.5 | toolret | query | 0.369 | 0.440 | 0.526 | 0.588 | 0.644 | 0.238 | +0.022 [+0.007, +0.035] | 109.482 | 56.765 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | expanded | query | 0.370 | 0.436 | 0.525 | 0.581 | 0.649 | 0.225 | +0.021 [+0.012, +0.030] | 22.713 | 9.148 |
| hybrid | LFM2.5-Embedding-350M | toolret | query | 0.363 | 0.427 | 0.514 | 0.566 | 0.627 | 0.238 | +0.010 [-0.002, +0.022] | 354.484 | 158.015 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | toolret | query | 0.348 | 0.424 | 0.504 | 0.560 | 0.618 | 0.226 | – | 22.713 | 9.148 |
| dense | LFM2.5-Embedding-350M | expanded | query | 0.343 | 0.406 | 0.490 | 0.557 | 0.613 | 0.191 | -0.014 [-0.030, +0.001] | 354.484 | 158.015 |
| dense | multi-qa-MiniLM-L6-cos-v1 | expanded | query | 0.332 | 0.384 | 0.469 | 0.538 | 0.577 | 0.206 | -0.035 [-0.049, -0.023] | 22.713 | 9.148 |
| dense | LFM2.5-Embedding-350M | toolret | query | 0.316 | 0.379 | 0.465 | 0.532 | 0.575 | 0.196 | -0.039 [-0.054, -0.023] | 354.484 | 158.015 |
| bm25 | – | expanded | query | 0.320 | 0.372 | 0.451 | 0.506 | 0.554 | 0.199 | -0.053 [-0.067, -0.041] | – | – |
| dense | multi-qa-MiniLM-L6-cos-v1 | toolret | query | 0.308 | 0.363 | 0.446 | 0.514 | 0.540 | 0.219 | -0.058 [-0.070, -0.047] | 22.713 | 9.148 |
| bm25 | – | toolret | query | 0.298 | 0.356 | 0.420 | 0.472 | 0.515 | 0.187 | -0.085 [-0.097, -0.072] | – | – |

## Recall@50 by catalog

| method | encoder | docs | query | code | customized |
|---|---|---|---|---|---|
| dense | Tool-Embed-0.6B | expanded | query | 0.658 | 0.479 |
| dense | Tool-Embed-0.6B | toolret | query | 0.644 | 0.482 |
| hybrid | Tool-Embed-0.6B | expanded | query | 0.660 | 0.453 |
| hybrid | Tool-Embed-0.6B | toolret | query | 0.655 | 0.459 |
| dense | ToolRet-trained-bge-base-en-v1.5 | expanded | query | 0.624 | 0.447 |
| hybrid | ToolRet-trained-bge-base-en-v1.5 | expanded | query | 0.620 | 0.447 |
| hybrid | ToolRet-trained-bge-base-en-v1.5 | toolret | query | 0.603 | 0.440 |
| hybrid | LFM2.5-Embedding-350M | expanded | query | 0.577 | 0.445 |
| dense | ToolRet-trained-bge-base-en-v1.5 | toolret | query | 0.581 | 0.427 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | expanded | query | 0.591 | 0.408 |
| hybrid | LFM2.5-Embedding-350M | toolret | query | 0.549 | 0.451 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | toolret | query | 0.559 | 0.406 |
| dense | LFM2.5-Embedding-350M | expanded | query | 0.524 | 0.430 |
| dense | multi-qa-MiniLM-L6-cos-v1 | expanded | query | 0.530 | 0.360 |
| dense | LFM2.5-Embedding-350M | toolret | query | 0.490 | 0.421 |
| bm25 | – | expanded | query | 0.508 | 0.348 |
| dense | multi-qa-MiniLM-L6-cos-v1 | toolret | query | 0.489 | 0.370 |
| bm25 | – | toolret | query | 0.478 | 0.316 |
