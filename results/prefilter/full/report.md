# Stage-1 prefilter benchmark (ToolRet)

7961 steps (4402 single-tool) over 44453 tools, catalogs: web, code, customized. Recall@K = all gold tools among the top K. `hybrid` = BM25 + dense fused with RRF (the current prefilter design). `expanded` docs = Tool-REX tool profiles. `query+instruction` uses ToolRet's target-aware instruction (written with the gold tools in view): an oracle upper bound only. ΔR@50 = paired bootstrap difference against the current prefilter (hybrid, MiniLM, ToolRet docs). Latency = CPU time to encode one query (4 cores, fp32, batch 16).

| method | encoder | docs | query | R@10 | R@20 | R@50 | R@100 | R@50 single | R@50 multi | ΔR@50 vs current [95% CI] | params (M) | query ms (CPU) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid | multi-qa-MiniLM-L6-cos-v1 | expanded | query | 0.365 | 0.432 | 0.528 | 0.587 | 0.672 | 0.351 | +0.036 [+0.030, +0.042] | 22.713 | 2.480 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | toolret | query | 0.338 | 0.408 | 0.493 | 0.553 | 0.648 | 0.301 | – | 22.713 | 2.480 |
| bm25 | – | expanded | query | 0.323 | 0.384 | 0.469 | 0.529 | 0.592 | 0.317 | -0.024 [-0.031, -0.016] | – | – |
| dense | multi-qa-MiniLM-L6-cos-v1 | expanded | query | 0.320 | 0.377 | 0.454 | 0.516 | 0.611 | 0.261 | -0.038 [-0.045, -0.031] | 22.713 | 2.480 |
| dense | multi-qa-MiniLM-L6-cos-v1 | toolret | query | 0.312 | 0.365 | 0.440 | 0.499 | 0.592 | 0.251 | -0.053 [-0.060, -0.046] | 22.713 | 2.480 |
| bm25 | – | toolret | query | 0.281 | 0.336 | 0.404 | 0.459 | 0.547 | 0.227 | -0.088 [-0.096, -0.081] | – | – |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | expanded | query+instruction | 0.471 | 0.543 | 0.635 | 0.695 | 0.805 | 0.425 | +0.142 [+0.134, +0.151] | 22.713 | 2.480 |
| bm25 | – | expanded | query+instruction | 0.456 | 0.529 | 0.614 | 0.671 | 0.778 | 0.410 | +0.121 [+0.112, +0.130] | – | – |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | toolret | query+instruction | 0.443 | 0.507 | 0.599 | 0.656 | 0.787 | 0.367 | +0.107 [+0.099, +0.115] | 22.713 | 2.480 |
| dense | multi-qa-MiniLM-L6-cos-v1 | expanded | query+instruction | 0.414 | 0.477 | 0.562 | 0.620 | 0.758 | 0.318 | +0.069 [+0.060, +0.078] | 22.713 | 2.480 |
| dense | multi-qa-MiniLM-L6-cos-v1 | toolret | query+instruction | 0.401 | 0.464 | 0.547 | 0.607 | 0.739 | 0.309 | +0.054 [+0.046, +0.063] | 22.713 | 2.480 |
| bm25 | – | toolret | query+instruction | 0.399 | 0.467 | 0.540 | 0.596 | 0.737 | 0.296 | +0.048 [+0.039, +0.056] | – | – |

## Recall@50 by catalog

| method | encoder | docs | query | web | code | customized |
|---|---|---|---|---|---|---|
| hybrid | multi-qa-MiniLM-L6-cos-v1 | expanded | query | 0.530 | 0.591 | 0.408 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | toolret | query | 0.486 | 0.559 | 0.406 |
| bm25 | – | expanded | query | 0.478 | 0.508 | 0.348 |
| dense | multi-qa-MiniLM-L6-cos-v1 | expanded | query | 0.447 | 0.530 | 0.360 |
| dense | multi-qa-MiniLM-L6-cos-v1 | toolret | query | 0.436 | 0.489 | 0.370 |
| bm25 | – | toolret | query | 0.396 | 0.478 | 0.316 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | expanded | query+instruction | 0.606 | 0.808 | 0.480 |
| bm25 | – | expanded | query+instruction | 0.590 | 0.771 | 0.459 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | toolret | query+instruction | 0.557 | 0.797 | 0.473 |
| dense | multi-qa-MiniLM-L6-cos-v1 | expanded | query+instruction | 0.526 | 0.739 | 0.434 |
| dense | multi-qa-MiniLM-L6-cos-v1 | toolret | query+instruction | 0.511 | 0.721 | 0.424 |
| bm25 | – | toolret | query+instruction | 0.496 | 0.740 | 0.422 |
