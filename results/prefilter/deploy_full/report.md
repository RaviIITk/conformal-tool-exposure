# Stage-1 prefilter benchmark (ToolRet)

7961 steps (4402 single-tool) over 44453 tools, catalogs: web, code, customized. Recall@K = all gold tools among the top K. `hybrid` = BM25 + dense fused with RRF (the current prefilter design). `expanded` docs = Tool-REX tool profiles. `query+instruction` uses ToolRet's target-aware instruction (written with the gold tools in view): an oracle upper bound only. Scope: `pooled` ranks the whole catalog; `routed<m>` ranks the tools of the m toolsets whose best tool scores highest first (hierarchical server-then-tool routing, no labels used); `toolset` ranks the step's own toolset first (oracle: the agent's connected toolset is known). Routing accuracy = the gold toolset is among the m routed ones. Match `duplicate`: a tool from the same toolset with the gold tool's exact description also counts (RoTBench ships renamed copies of each tool). An encoder `A + B` fuses BM25 and both encoders with RRF. ΔR@50 = paired bootstrap difference against the current prefilter (hybrid, MiniLM, ToolRet docs, pooled). Latency = CPU time to encode one query (4 cores, fp32, batch 16).

| method | encoder | docs | query | scope | match | R@10 | R@20 | R@50 | R@100 | R@200 | R@20 single | R@50 single | R@100 single | R@50 multi | routing acc. | ΔR@50 vs current [95% CI] | params (M) | query ms (CPU) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid | multi-qa-MiniLM-L6-cos-v1 | expanded | query | toolset | duplicate | 0.531 | 0.618 | 0.720 | 0.798 | 0.861 | 0.753 | 0.838 | 0.905 | 0.574 | – | +0.227 [+0.218, +0.237] | 22.713 | 6.905 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | expanded | query | toolset | exact | 0.515 | 0.603 | 0.703 | 0.770 | 0.834 | 0.738 | 0.818 | 0.866 | 0.561 | – | +0.210 [+0.201, +0.220] | 22.713 | 6.905 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | toolret | query | toolset | duplicate | 0.505 | 0.588 | 0.698 | 0.774 | 0.836 | 0.734 | 0.827 | 0.880 | 0.538 | – | +0.205 [+0.196, +0.214] | 22.713 | 6.905 |
| dense | multi-qa-MiniLM-L6-cos-v1 | expanded | query | toolset | duplicate | 0.477 | 0.563 | 0.682 | 0.763 | 0.833 | 0.718 | 0.827 | 0.896 | 0.502 | – | +0.189 [+0.179, +0.200] | 22.713 | 6.905 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | toolret | query | toolset | exact | 0.493 | 0.577 | 0.681 | 0.752 | 0.812 | 0.722 | 0.804 | 0.853 | 0.528 | – | +0.188 [+0.180, +0.197] | 22.713 | 6.905 |
| dense | multi-qa-MiniLM-L6-cos-v1 | toolret | query | toolset | duplicate | 0.468 | 0.551 | 0.675 | 0.754 | 0.819 | 0.714 | 0.819 | 0.876 | 0.496 | – | +0.182 [+0.172, +0.192] | 22.713 | 6.905 |
| dense | multi-qa-MiniLM-L6-cos-v1 | expanded | query | toolset | exact | 0.458 | 0.548 | 0.661 | 0.735 | 0.805 | 0.703 | 0.804 | 0.853 | 0.484 | – | +0.168 [+0.159, +0.178] | 22.713 | 6.905 |
| dense | multi-qa-MiniLM-L6-cos-v1 | toolret | query | toolset | exact | 0.454 | 0.536 | 0.654 | 0.731 | 0.795 | 0.698 | 0.792 | 0.844 | 0.483 | – | +0.161 [+0.152, +0.171] | 22.713 | 6.905 |
| bm25 | – | expanded | query | toolset | duplicate | 0.480 | 0.559 | 0.636 | 0.692 | 0.734 | 0.675 | 0.739 | 0.781 | 0.510 | – | +0.144 [+0.134, +0.153] | – | – |
| bm25 | – | expanded | query | toolset | exact | 0.464 | 0.544 | 0.620 | 0.674 | 0.713 | 0.665 | 0.725 | 0.763 | 0.490 | – | +0.127 [+0.118, +0.137] | – | – |
| bm25 | – | toolret | query | toolset | duplicate | 0.423 | 0.485 | 0.559 | 0.603 | 0.635 | 0.627 | 0.690 | 0.723 | 0.398 | – | +0.067 [+0.058, +0.075] | – | – |
| bm25 | – | toolret | query | toolset | exact | 0.415 | 0.480 | 0.555 | 0.598 | 0.629 | 0.624 | 0.688 | 0.723 | 0.391 | – | +0.062 [+0.054, +0.071] | – | – |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | expanded | query | pooled | duplicate | 0.378 | 0.447 | 0.540 | 0.598 | 0.650 | 0.592 | 0.677 | 0.729 | 0.370 | – | +0.047 [+0.041, +0.054] | 22.713 | 6.905 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | expanded | query | routed3 | duplicate | 0.376 | 0.444 | 0.529 | 0.583 | 0.631 | 0.583 | 0.653 | 0.701 | 0.375 | 0.857 | +0.036 [+0.030, +0.043] | 22.713 | 6.905 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | expanded | query | pooled | exact | 0.365 | 0.432 | 0.528 | 0.587 | 0.638 | 0.583 | 0.672 | 0.721 | 0.351 | – | +0.036 [+0.030, +0.042] | 22.713 | 6.905 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | expanded | query | routed3 | exact | 0.363 | 0.431 | 0.518 | 0.573 | 0.619 | 0.574 | 0.647 | 0.693 | 0.358 | 0.857 | +0.025 [+0.019, +0.032] | 22.713 | 6.905 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | toolret | query | pooled | duplicate | 0.345 | 0.416 | 0.501 | 0.563 | 0.617 | 0.570 | 0.651 | 0.708 | 0.316 | – | +0.008 [+0.007, +0.010] | 22.713 | 6.905 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | toolret | query | pooled | exact | 0.338 | 0.408 | 0.493 | 0.553 | 0.606 | 0.564 | 0.648 | 0.702 | 0.301 | – | – | 22.713 | 6.905 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | toolret | query | routed3 | duplicate | 0.345 | 0.411 | 0.492 | 0.545 | 0.590 | 0.558 | 0.629 | 0.675 | 0.323 | 0.839 | -0.001 [-0.005, +0.004] | 22.713 | 6.905 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | toolret | query | routed3 | exact | 0.338 | 0.404 | 0.486 | 0.538 | 0.582 | 0.554 | 0.625 | 0.671 | 0.314 | 0.839 | -0.007 [-0.011, -0.003] | 22.713 | 6.905 |
| bm25 | – | expanded | query | pooled | duplicate | 0.336 | 0.398 | 0.483 | 0.542 | 0.600 | 0.522 | 0.598 | 0.654 | 0.341 | – | -0.009 [-0.018, -0.001] | – | – |
| bm25 | – | expanded | query | routed3 | duplicate | 0.335 | 0.395 | 0.475 | 0.525 | 0.571 | 0.511 | 0.574 | 0.616 | 0.352 | 0.833 | -0.018 [-0.026, -0.009] | – | – |
| dense | multi-qa-MiniLM-L6-cos-v1 | expanded | query | pooled | duplicate | 0.329 | 0.386 | 0.469 | 0.530 | 0.595 | 0.531 | 0.622 | 0.682 | 0.281 | – | -0.023 [-0.031, -0.015] | 22.713 | 6.905 |
| bm25 | – | expanded | query | pooled | exact | 0.323 | 0.384 | 0.469 | 0.529 | 0.586 | 0.514 | 0.592 | 0.647 | 0.317 | – | -0.024 [-0.031, -0.016] | – | – |
| bm25 | – | expanded | query | routed3 | exact | 0.323 | 0.381 | 0.463 | 0.515 | 0.560 | 0.504 | 0.569 | 0.612 | 0.331 | 0.833 | -0.030 [-0.038, -0.022] | – | – |
| dense | multi-qa-MiniLM-L6-cos-v1 | expanded | query | routed3 | duplicate | 0.326 | 0.381 | 0.462 | 0.515 | 0.567 | 0.519 | 0.604 | 0.653 | 0.287 | 0.836 | -0.030 [-0.038, -0.022] | 22.713 | 6.905 |
| dense | multi-qa-MiniLM-L6-cos-v1 | expanded | query | pooled | exact | 0.320 | 0.377 | 0.454 | 0.516 | 0.581 | 0.527 | 0.611 | 0.671 | 0.261 | – | -0.038 [-0.045, -0.031] | 22.713 | 6.905 |
| dense | multi-qa-MiniLM-L6-cos-v1 | toolret | query | pooled | duplicate | 0.321 | 0.372 | 0.451 | 0.512 | 0.579 | 0.517 | 0.602 | 0.665 | 0.264 | – | -0.042 [-0.049, -0.035] | 22.713 | 6.905 |
| dense | multi-qa-MiniLM-L6-cos-v1 | expanded | query | routed3 | exact | 0.318 | 0.372 | 0.448 | 0.503 | 0.555 | 0.515 | 0.592 | 0.641 | 0.269 | 0.836 | -0.045 [-0.053, -0.037] | 22.713 | 6.905 |
| dense | multi-qa-MiniLM-L6-cos-v1 | toolret | query | pooled | exact | 0.312 | 0.365 | 0.440 | 0.499 | 0.566 | 0.512 | 0.592 | 0.655 | 0.251 | – | -0.053 [-0.060, -0.046] | 22.713 | 6.905 |
| dense | multi-qa-MiniLM-L6-cos-v1 | toolret | query | routed3 | duplicate | 0.317 | 0.363 | 0.438 | 0.491 | 0.545 | 0.500 | 0.576 | 0.624 | 0.266 | 0.821 | -0.055 [-0.062, -0.047] | 22.713 | 6.905 |
| dense | multi-qa-MiniLM-L6-cos-v1 | toolret | query | routed3 | exact | 0.309 | 0.357 | 0.428 | 0.480 | 0.536 | 0.497 | 0.568 | 0.616 | 0.255 | 0.821 | -0.064 [-0.072, -0.057] | 22.713 | 6.905 |
| bm25 | – | toolret | query | pooled | duplicate | 0.286 | 0.341 | 0.407 | 0.464 | 0.516 | 0.480 | 0.548 | 0.602 | 0.233 | – | -0.085 [-0.093, -0.078] | – | – |
| bm25 | – | toolret | query | pooled | exact | 0.281 | 0.336 | 0.404 | 0.459 | 0.510 | 0.477 | 0.547 | 0.600 | 0.227 | – | -0.088 [-0.096, -0.081] | – | – |
| bm25 | – | toolret | query | routed3 | duplicate | 0.281 | 0.333 | 0.396 | 0.443 | 0.486 | 0.460 | 0.518 | 0.557 | 0.245 | 0.799 | -0.096 [-0.105, -0.089] | – | – |
| bm25 | – | toolret | query | routed3 | exact | 0.277 | 0.328 | 0.393 | 0.439 | 0.482 | 0.457 | 0.518 | 0.557 | 0.238 | 0.799 | -0.099 [-0.108, -0.092] | – | – |

## Recall@50 by catalog

| method | encoder | docs | query | scope | match | web | code | customized |
|---|---|---|---|---|---|---|---|---|
| hybrid | multi-qa-MiniLM-L6-cos-v1 | expanded | query | toolset | duplicate | 0.743 | 0.718 | 0.598 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | expanded | query | toolset | exact | 0.720 | 0.712 | 0.598 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | toolret | query | toolset | duplicate | 0.713 | 0.709 | 0.594 |
| dense | multi-qa-MiniLM-L6-cos-v1 | expanded | query | toolset | duplicate | 0.692 | 0.696 | 0.600 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | toolret | query | toolset | exact | 0.688 | 0.707 | 0.594 |
| dense | multi-qa-MiniLM-L6-cos-v1 | toolret | query | toolset | duplicate | 0.683 | 0.692 | 0.601 |
| dense | multi-qa-MiniLM-L6-cos-v1 | expanded | query | toolset | exact | 0.661 | 0.695 | 0.600 |
| dense | multi-qa-MiniLM-L6-cos-v1 | toolret | query | toolset | exact | 0.652 | 0.689 | 0.601 |
| bm25 | – | expanded | query | toolset | duplicate | 0.672 | 0.631 | 0.452 |
| bm25 | – | expanded | query | toolset | exact | 0.650 | 0.623 | 0.452 |
| bm25 | – | toolret | query | toolset | duplicate | 0.575 | 0.588 | 0.424 |
| bm25 | – | toolret | query | toolset | exact | 0.568 | 0.588 | 0.424 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | expanded | query | pooled | duplicate | 0.546 | 0.595 | 0.408 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | expanded | query | routed3 | duplicate | 0.529 | 0.589 | 0.421 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | expanded | query | pooled | exact | 0.530 | 0.591 | 0.408 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | expanded | query | routed3 | exact | 0.513 | 0.586 | 0.421 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | toolret | query | pooled | duplicate | 0.499 | 0.561 | 0.406 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | toolret | query | pooled | exact | 0.486 | 0.559 | 0.406 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | toolret | query | routed3 | duplicate | 0.487 | 0.557 | 0.404 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | toolret | query | routed3 | exact | 0.478 | 0.556 | 0.404 |
| bm25 | – | expanded | query | pooled | duplicate | 0.498 | 0.515 | 0.348 |
| bm25 | – | expanded | query | routed3 | duplicate | 0.489 | 0.506 | 0.344 |
| dense | multi-qa-MiniLM-L6-cos-v1 | expanded | query | pooled | duplicate | 0.468 | 0.533 | 0.360 |
| bm25 | – | expanded | query | pooled | exact | 0.478 | 0.508 | 0.348 |
| bm25 | – | expanded | query | routed3 | exact | 0.473 | 0.500 | 0.344 |
| dense | multi-qa-MiniLM-L6-cos-v1 | expanded | query | routed3 | duplicate | 0.456 | 0.531 | 0.374 |
| dense | multi-qa-MiniLM-L6-cos-v1 | expanded | query | pooled | exact | 0.447 | 0.530 | 0.360 |
| dense | multi-qa-MiniLM-L6-cos-v1 | toolret | query | pooled | duplicate | 0.452 | 0.492 | 0.370 |
| dense | multi-qa-MiniLM-L6-cos-v1 | expanded | query | routed3 | exact | 0.434 | 0.528 | 0.374 |
| dense | multi-qa-MiniLM-L6-cos-v1 | toolret | query | pooled | exact | 0.436 | 0.489 | 0.370 |
| dense | multi-qa-MiniLM-L6-cos-v1 | toolret | query | routed3 | duplicate | 0.431 | 0.491 | 0.378 |
| dense | multi-qa-MiniLM-L6-cos-v1 | toolret | query | routed3 | exact | 0.417 | 0.489 | 0.378 |
| bm25 | – | toolret | query | pooled | duplicate | 0.401 | 0.478 | 0.316 |
| bm25 | – | toolret | query | pooled | exact | 0.396 | 0.478 | 0.316 |
| bm25 | – | toolret | query | routed3 | duplicate | 0.387 | 0.471 | 0.311 |
| bm25 | – | toolret | query | routed3 | exact | 0.383 | 0.471 | 0.311 |
