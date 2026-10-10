# Stage-1 prefilter benchmark (ToolRet)

2731 steps (1936 single-tool) over 7161 tools, catalogs: code, customized. Recall@K = all gold tools among the top K. `hybrid` = BM25 + dense fused with RRF (the current prefilter design). `expanded` docs = Tool-REX tool profiles. `query+instruction` uses ToolRet's target-aware instruction (written with the gold tools in view): an oracle upper bound only. Scope: `pooled` ranks the whole catalog; `routed<m>` ranks the tools of the m toolsets whose best tool scores highest first (hierarchical server-then-tool routing, no labels used); `toolset` ranks the step's own toolset first (oracle: the agent's connected toolset is known). Routing accuracy = the gold toolset is among the m routed ones. Match `duplicate`: a tool from the same toolset with the gold tool's exact description also counts (RoTBench ships renamed copies of each tool). An encoder `A + B` fuses BM25 and both encoders with RRF. ΔR@50 = paired bootstrap difference against the current prefilter (hybrid, MiniLM, ToolRet docs, pooled). Latency = CPU time to encode one query (4 cores, fp32, batch 16).

| method | encoder | docs | query | scope | match | R@10 | R@20 | R@50 | R@100 | R@200 | R@20 single | R@50 single | R@100 single | R@50 multi | routing acc. | ΔR@50 vs current [95% CI] | params (M) | query ms (CPU) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | expanded | query | toolset | duplicate | 0.568 | 0.638 | 0.755 | 0.797 | 0.834 | 0.791 | 0.900 | 0.948 | 0.401 | – | +0.251 [+0.234, +0.269] | 705.259 | 450.499 |
| dense | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | expanded | query | toolset | exact | 0.563 | 0.635 | 0.754 | 0.796 | 0.833 | 0.787 | 0.899 | 0.947 | 0.401 | – | +0.250 [+0.234, +0.267] | 705.259 | 450.499 |
| hybrid | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | expanded | query | toolset | duplicate | 0.556 | 0.635 | 0.753 | 0.799 | 0.831 | 0.787 | 0.898 | 0.942 | 0.400 | – | +0.249 [+0.233, +0.266] | 705.259 | 450.499 |
| hybrid | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | expanded | query | toolset | exact | 0.549 | 0.631 | 0.752 | 0.799 | 0.830 | 0.781 | 0.896 | 0.941 | 0.400 | – | +0.248 [+0.232, +0.265] | 705.259 | 450.499 |
| hybrid | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | toolret | query | toolset | duplicate | 0.551 | 0.643 | 0.744 | 0.787 | 0.821 | 0.762 | 0.874 | 0.919 | 0.426 | – | +0.240 [+0.224, +0.257] | 705.259 | 450.499 |
| dense | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | toolret | query | toolset | duplicate | 0.552 | 0.648 | 0.744 | 0.795 | 0.822 | 0.769 | 0.873 | 0.927 | 0.428 | – | +0.239 [+0.223, +0.257] | 705.259 | 450.499 |
| dense | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | toolret | query | toolset | exact | 0.550 | 0.646 | 0.742 | 0.792 | 0.821 | 0.765 | 0.871 | 0.924 | 0.428 | – | +0.238 [+0.221, +0.256] | 705.259 | 450.499 |
| hybrid | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | toolret | query | toolset | exact | 0.550 | 0.640 | 0.742 | 0.785 | 0.820 | 0.759 | 0.871 | 0.916 | 0.426 | – | +0.238 [+0.222, +0.254] | 705.259 | 450.499 |
| dense | Tool-Embed-0.6B | expanded | query | toolset | duplicate | 0.534 | 0.632 | 0.741 | 0.786 | 0.830 | 0.754 | 0.879 | 0.922 | 0.406 | – | +0.237 [+0.220, +0.256] | 595.777 | 372.148 |
| dense | Tool-Embed-0.6B | expanded | query | toolset | exact | 0.528 | 0.629 | 0.739 | 0.784 | 0.830 | 0.749 | 0.876 | 0.919 | 0.405 | – | +0.234 [+0.217, +0.253] | 595.777 | 372.148 |
| hybrid | Tool-Embed-0.6B | expanded | query | toolset | duplicate | 0.532 | 0.634 | 0.731 | 0.785 | 0.832 | 0.755 | 0.871 | 0.926 | 0.389 | – | +0.226 [+0.210, +0.244] | 595.777 | 372.148 |
| hybrid | Tool-Embed-0.6B | toolret | query | toolset | duplicate | 0.537 | 0.624 | 0.729 | 0.770 | 0.810 | 0.737 | 0.851 | 0.895 | 0.434 | – | +0.225 [+0.209, +0.242] | 595.777 | 372.148 |
| hybrid | Tool-Embed-0.6B | toolret | query | toolset | exact | 0.534 | 0.622 | 0.728 | 0.769 | 0.810 | 0.735 | 0.849 | 0.893 | 0.434 | – | +0.224 [+0.208, +0.241] | 595.777 | 372.148 |
| dense | Tool-Embed-0.6B | toolret | query | toolset | duplicate | 0.551 | 0.631 | 0.728 | 0.770 | 0.805 | 0.733 | 0.841 | 0.895 | 0.452 | – | +0.223 [+0.206, +0.242] | 595.777 | 372.148 |
| hybrid | Tool-Embed-0.6B | expanded | query | toolset | exact | 0.525 | 0.630 | 0.727 | 0.784 | 0.831 | 0.749 | 0.866 | 0.924 | 0.389 | – | +0.223 [+0.207, +0.240] | 595.777 | 372.148 |
| dense | Tool-Embed-0.6B | toolret | query | toolset | exact | 0.549 | 0.629 | 0.726 | 0.769 | 0.804 | 0.731 | 0.839 | 0.894 | 0.452 | – | +0.222 [+0.205, +0.241] | 595.777 | 372.148 |
| dense | ToolRet-trained-bge-base-en-v1.5 | expanded | query | toolset | duplicate | 0.533 | 0.618 | 0.692 | 0.750 | 0.792 | 0.766 | 0.857 | 0.927 | 0.289 | – | +0.188 [+0.172, +0.204] | 109.482 | 52.471 |
| dense | ToolRet-trained-bge-base-en-v1.5 | expanded | query | toolset | exact | 0.527 | 0.615 | 0.691 | 0.748 | 0.792 | 0.762 | 0.856 | 0.926 | 0.289 | – | +0.187 [+0.171, +0.202] | 109.482 | 52.471 |
| hybrid | ToolRet-trained-bge-base-en-v1.5 | expanded | query | toolset | duplicate | 0.528 | 0.601 | 0.685 | 0.742 | 0.789 | 0.740 | 0.849 | 0.919 | 0.284 | – | +0.181 [+0.166, +0.196] | 109.482 | 52.471 |
| hybrid | ToolRet-trained-bge-base-en-v1.5 | expanded | query | toolset | exact | 0.520 | 0.595 | 0.682 | 0.742 | 0.788 | 0.733 | 0.845 | 0.918 | 0.284 | – | +0.178 [+0.163, +0.193] | 109.482 | 52.471 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | expanded | query | toolset | duplicate | 0.499 | 0.574 | 0.675 | 0.731 | 0.795 | 0.705 | 0.835 | 0.897 | 0.284 | – | +0.171 [+0.156, +0.185] | 22.713 | 9.039 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | expanded | query | toolset | exact | 0.493 | 0.570 | 0.671 | 0.730 | 0.793 | 0.699 | 0.830 | 0.895 | 0.284 | – | +0.167 [+0.153, +0.182] | 22.713 | 9.039 |
| dense | ToolRet-trained-bge-base-en-v1.5 | toolret | query | toolset | duplicate | 0.513 | 0.592 | 0.670 | 0.723 | 0.777 | 0.733 | 0.830 | 0.896 | 0.281 | – | +0.166 [+0.150, +0.182] | 109.482 | 52.471 |
| dense | ToolRet-trained-bge-base-en-v1.5 | toolret | query | toolset | exact | 0.510 | 0.590 | 0.668 | 0.722 | 0.775 | 0.729 | 0.827 | 0.894 | 0.281 | – | +0.164 [+0.149, +0.180] | 109.482 | 52.471 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | toolret | query | toolset | duplicate | 0.479 | 0.558 | 0.668 | 0.729 | 0.778 | 0.683 | 0.806 | 0.867 | 0.331 | – | +0.163 [+0.149, +0.178] | 22.713 | 9.039 |
| hybrid | ToolRet-trained-bge-base-en-v1.5 | toolret | query | toolset | duplicate | 0.509 | 0.587 | 0.666 | 0.717 | 0.773 | 0.724 | 0.823 | 0.887 | 0.286 | – | +0.162 [+0.148, +0.177] | 109.482 | 52.471 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | toolret | query | toolset | exact | 0.476 | 0.557 | 0.666 | 0.728 | 0.777 | 0.681 | 0.804 | 0.866 | 0.331 | – | +0.162 [+0.148, +0.176] | 22.713 | 9.039 |
| hybrid | ToolRet-trained-bge-base-en-v1.5 | toolret | query | toolset | exact | 0.506 | 0.585 | 0.665 | 0.716 | 0.772 | 0.721 | 0.821 | 0.886 | 0.286 | – | +0.161 [+0.147, +0.176] | 109.482 | 52.471 |
| dense | multi-qa-MiniLM-L6-cos-v1 | expanded | query | toolset | duplicate | 0.455 | 0.543 | 0.662 | 0.730 | 0.794 | 0.667 | 0.817 | 0.886 | 0.283 | – | +0.157 [+0.143, +0.173] | 22.713 | 9.039 |
| dense | multi-qa-MiniLM-L6-cos-v1 | expanded | query | toolset | exact | 0.452 | 0.541 | 0.661 | 0.729 | 0.793 | 0.664 | 0.816 | 0.884 | 0.283 | – | +0.156 [+0.142, +0.172] | 22.713 | 9.039 |
| dense | multi-qa-MiniLM-L6-cos-v1 | toolret | query | toolset | duplicate | 0.439 | 0.537 | 0.659 | 0.721 | 0.778 | 0.658 | 0.793 | 0.855 | 0.332 | – | +0.155 [+0.140, +0.170] | 22.713 | 9.039 |
| dense | multi-qa-MiniLM-L6-cos-v1 | toolret | query | toolset | exact | 0.436 | 0.536 | 0.657 | 0.720 | 0.777 | 0.657 | 0.791 | 0.853 | 0.332 | – | +0.153 [+0.138, +0.168] | 22.713 | 9.039 |
| dense | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | expanded | query | routed3 | duplicate | 0.436 | 0.501 | 0.612 | 0.676 | 0.716 | 0.619 | 0.723 | 0.794 | 0.342 | 0.941 | +0.108 [+0.092, +0.125] | 705.259 | 450.499 |
| dense | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | expanded | query | routed3 | exact | 0.430 | 0.498 | 0.610 | 0.675 | 0.715 | 0.615 | 0.721 | 0.792 | 0.342 | 0.941 | +0.106 [+0.090, +0.123] | 705.259 | 450.499 |
| dense | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | expanded | query | pooled | duplicate | 0.438 | 0.500 | 0.610 | 0.680 | 0.733 | 0.621 | 0.721 | 0.799 | 0.341 | – | +0.106 [+0.090, +0.122] | 705.259 | 450.499 |
| dense | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | expanded | query | pooled | exact | 0.432 | 0.497 | 0.608 | 0.679 | 0.732 | 0.617 | 0.718 | 0.797 | 0.341 | – | +0.104 [+0.089, +0.119] | 705.259 | 450.499 |
| hybrid | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | toolret | query | pooled | duplicate | 0.427 | 0.498 | 0.607 | 0.668 | 0.725 | 0.617 | 0.713 | 0.777 | 0.350 | – | +0.103 [+0.089, +0.116] | 705.259 | 450.499 |
| hybrid | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | toolret | query | pooled | exact | 0.425 | 0.496 | 0.604 | 0.666 | 0.723 | 0.614 | 0.709 | 0.774 | 0.350 | – | +0.100 [+0.086, +0.113] | 705.259 | 450.499 |
| hybrid | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | toolret | query | routed3 | duplicate | 0.425 | 0.498 | 0.604 | 0.674 | 0.715 | 0.616 | 0.704 | 0.780 | 0.360 | 0.938 | +0.100 [+0.086, +0.113] | 705.259 | 450.499 |
| hybrid | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | toolret | query | routed3 | exact | 0.423 | 0.496 | 0.601 | 0.673 | 0.714 | 0.613 | 0.700 | 0.778 | 0.360 | 0.938 | +0.097 [+0.083, +0.111] | 705.259 | 450.499 |
| dense | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | toolret | query | pooled | duplicate | 0.418 | 0.487 | 0.597 | 0.667 | 0.729 | 0.600 | 0.693 | 0.770 | 0.365 | – | +0.093 [+0.078, +0.108] | 705.259 | 450.499 |
| dense | Tool-Embed-0.6B | expanded | query | routed3 | duplicate | 0.403 | 0.492 | 0.597 | 0.648 | 0.696 | 0.579 | 0.702 | 0.756 | 0.341 | 0.933 | +0.093 [+0.075, +0.111] | 595.777 | 372.148 |
| dense | Tool-Embed-0.6B | expanded | query | pooled | duplicate | 0.405 | 0.490 | 0.596 | 0.653 | 0.709 | 0.577 | 0.704 | 0.762 | 0.336 | – | +0.092 [+0.075, +0.110] | 595.777 | 372.148 |
| hybrid | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | expanded | query | routed3 | duplicate | 0.434 | 0.508 | 0.596 | 0.687 | 0.731 | 0.631 | 0.735 | 0.802 | 0.258 | 0.944 | +0.092 [+0.078, +0.105] | 705.259 | 450.499 |
| dense | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | toolret | query | pooled | exact | 0.416 | 0.485 | 0.595 | 0.665 | 0.728 | 0.597 | 0.690 | 0.767 | 0.365 | – | +0.091 [+0.076, +0.106] | 705.259 | 450.499 |
| hybrid | Tool-Embed-0.6B | expanded | query | routed3 | duplicate | 0.415 | 0.481 | 0.595 | 0.649 | 0.700 | 0.595 | 0.704 | 0.766 | 0.328 | 0.928 | +0.090 [+0.075, +0.106] | 595.777 | 372.148 |
| hybrid | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | expanded | query | pooled | duplicate | 0.436 | 0.506 | 0.594 | 0.681 | 0.744 | 0.630 | 0.734 | 0.798 | 0.254 | – | +0.090 [+0.077, +0.103] | 705.259 | 450.499 |
| hybrid | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | expanded | query | routed3 | exact | 0.427 | 0.504 | 0.594 | 0.686 | 0.730 | 0.626 | 0.732 | 0.801 | 0.258 | 0.944 | +0.090 [+0.076, +0.103] | 705.259 | 450.499 |
| dense | Tool-Embed-0.6B | expanded | query | routed3 | exact | 0.397 | 0.489 | 0.594 | 0.646 | 0.696 | 0.575 | 0.698 | 0.753 | 0.340 | 0.933 | +0.089 [+0.072, +0.107] | 595.777 | 372.148 |
| dense | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | toolret | query | routed3 | duplicate | 0.417 | 0.486 | 0.594 | 0.663 | 0.708 | 0.598 | 0.689 | 0.763 | 0.362 | 0.934 | +0.089 [+0.073, +0.105] | 705.259 | 450.499 |
| dense | Tool-Embed-0.6B | expanded | query | pooled | exact | 0.399 | 0.486 | 0.593 | 0.651 | 0.709 | 0.572 | 0.699 | 0.759 | 0.335 | – | +0.089 [+0.072, +0.106] | 595.777 | 372.148 |
| hybrid | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | expanded | query | pooled | exact | 0.430 | 0.502 | 0.592 | 0.680 | 0.744 | 0.624 | 0.731 | 0.796 | 0.254 | – | +0.088 [+0.075, +0.101] | 705.259 | 450.499 |
| dense | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | toolret | query | routed3 | exact | 0.414 | 0.483 | 0.592 | 0.661 | 0.707 | 0.595 | 0.686 | 0.760 | 0.362 | 0.934 | +0.088 [+0.071, +0.103] | 705.259 | 450.499 |
| hybrid | Tool-Embed-0.6B | expanded | query | routed3 | exact | 0.408 | 0.477 | 0.591 | 0.647 | 0.699 | 0.590 | 0.699 | 0.763 | 0.328 | 0.928 | +0.087 [+0.073, +0.103] | 595.777 | 372.148 |
| hybrid | Tool-Embed-0.6B | toolret | query | routed3 | duplicate | 0.401 | 0.491 | 0.590 | 0.651 | 0.690 | 0.574 | 0.683 | 0.747 | 0.365 | 0.927 | +0.086 [+0.072, +0.101] | 595.777 | 372.148 |
| dense | Tool-Embed-0.6B | toolret | query | routed3 | duplicate | 0.417 | 0.490 | 0.590 | 0.639 | 0.690 | 0.571 | 0.677 | 0.729 | 0.379 | 0.934 | +0.086 [+0.068, +0.103] | 595.777 | 372.148 |
| hybrid | Tool-Embed-0.6B | expanded | query | pooled | duplicate | 0.416 | 0.480 | 0.589 | 0.656 | 0.716 | 0.595 | 0.698 | 0.777 | 0.325 | – | +0.085 [+0.071, +0.100] | 595.777 | 372.148 |
| hybrid | Tool-Embed-0.6B | toolret | query | routed3 | exact | 0.399 | 0.490 | 0.589 | 0.649 | 0.689 | 0.572 | 0.681 | 0.745 | 0.365 | 0.927 | +0.085 [+0.071, +0.099] | 595.777 | 372.148 |
| dense | Tool-Embed-0.6B | toolret | query | routed3 | exact | 0.415 | 0.488 | 0.588 | 0.637 | 0.688 | 0.568 | 0.675 | 0.727 | 0.379 | 0.934 | +0.084 [+0.067, +0.101] | 595.777 | 372.148 |
| dense | Tool-Embed-0.6B | toolret | query | pooled | duplicate | 0.417 | 0.486 | 0.587 | 0.639 | 0.702 | 0.568 | 0.672 | 0.729 | 0.380 | – | +0.083 [+0.066, +0.100] | 595.777 | 372.148 |
| dense | Tool-Embed-0.6B | toolret | query | pooled | exact | 0.415 | 0.484 | 0.585 | 0.637 | 0.699 | 0.565 | 0.670 | 0.726 | 0.380 | – | +0.081 [+0.064, +0.098] | 595.777 | 372.148 |
| hybrid | Tool-Embed-0.6B | toolret | query | pooled | duplicate | 0.402 | 0.491 | 0.585 | 0.652 | 0.705 | 0.576 | 0.678 | 0.751 | 0.361 | – | +0.081 [+0.068, +0.095] | 595.777 | 372.148 |
| hybrid | Tool-Embed-0.6B | expanded | query | pooled | exact | 0.409 | 0.476 | 0.585 | 0.654 | 0.715 | 0.589 | 0.693 | 0.775 | 0.325 | – | +0.081 [+0.067, +0.096] | 595.777 | 372.148 |
| hybrid | Tool-Embed-0.6B | toolret | query | pooled | exact | 0.399 | 0.490 | 0.584 | 0.650 | 0.704 | 0.575 | 0.676 | 0.748 | 0.361 | – | +0.080 [+0.067, +0.094] | 595.777 | 372.148 |
| bm25 | – | expanded | query | toolset | duplicate | 0.439 | 0.504 | 0.567 | 0.599 | 0.624 | 0.616 | 0.694 | 0.736 | 0.257 | – | +0.063 [+0.049, +0.077] | – | – |
| bm25 | – | expanded | query | toolset | exact | 0.433 | 0.499 | 0.562 | 0.595 | 0.621 | 0.608 | 0.687 | 0.731 | 0.257 | – | +0.057 [+0.044, +0.071] | – | – |
| dense | ToolRet-trained-bge-base-en-v1.5 | expanded | query | pooled | duplicate | 0.394 | 0.473 | 0.562 | 0.622 | 0.669 | 0.586 | 0.691 | 0.764 | 0.248 | – | +0.057 [+0.043, +0.072] | 109.482 | 52.471 |
| hybrid | ToolRet-trained-bge-base-en-v1.5 | expanded | query | routed3 | duplicate | 0.398 | 0.468 | 0.562 | 0.616 | 0.656 | 0.579 | 0.693 | 0.758 | 0.242 | 0.934 | +0.057 [+0.044, +0.070] | 109.482 | 52.471 |
| hybrid | ToolRet-trained-bge-base-en-v1.5 | expanded | query | pooled | duplicate | 0.398 | 0.466 | 0.561 | 0.620 | 0.670 | 0.579 | 0.694 | 0.764 | 0.238 | – | +0.057 [+0.044, +0.069] | 109.482 | 52.471 |
| dense | ToolRet-trained-bge-base-en-v1.5 | expanded | query | pooled | exact | 0.389 | 0.470 | 0.560 | 0.621 | 0.668 | 0.582 | 0.689 | 0.762 | 0.248 | – | +0.056 [+0.042, +0.070] | 109.482 | 52.471 |
| hybrid | ToolRet-trained-bge-base-en-v1.5 | expanded | query | routed3 | exact | 0.391 | 0.462 | 0.559 | 0.614 | 0.655 | 0.571 | 0.690 | 0.755 | 0.242 | 0.934 | +0.055 [+0.042, +0.067] | 109.482 | 52.471 |
| hybrid | ToolRet-trained-bge-base-en-v1.5 | expanded | query | pooled | exact | 0.391 | 0.461 | 0.558 | 0.618 | 0.669 | 0.571 | 0.690 | 0.761 | 0.238 | – | +0.054 [+0.041, +0.066] | 109.482 | 52.471 |
| dense | ToolRet-trained-bge-base-en-v1.5 | expanded | query | routed3 | duplicate | 0.397 | 0.475 | 0.557 | 0.608 | 0.647 | 0.585 | 0.684 | 0.744 | 0.247 | 0.923 | +0.053 [+0.038, +0.068] | 109.482 | 52.471 |
| dense | ToolRet-trained-bge-base-en-v1.5 | expanded | query | routed3 | exact | 0.391 | 0.472 | 0.555 | 0.607 | 0.646 | 0.581 | 0.682 | 0.743 | 0.247 | 0.923 | +0.051 [+0.037, +0.066] | 109.482 | 52.471 |
| hybrid | ToolRet-trained-bge-base-en-v1.5 | toolret | query | pooled | duplicate | 0.386 | 0.462 | 0.545 | 0.594 | 0.644 | 0.572 | 0.674 | 0.728 | 0.231 | – | +0.041 [+0.030, +0.052] | 109.482 | 52.471 |
| hybrid | ToolRet-trained-bge-base-en-v1.5 | toolret | query | pooled | exact | 0.384 | 0.460 | 0.544 | 0.592 | 0.643 | 0.569 | 0.673 | 0.726 | 0.231 | – | +0.040 [+0.029, +0.051] | 109.482 | 52.471 |
| hybrid | ToolRet-trained-bge-base-en-v1.5 | toolret | query | routed3 | duplicate | 0.385 | 0.462 | 0.539 | 0.584 | 0.628 | 0.569 | 0.663 | 0.711 | 0.239 | 0.924 | +0.035 [+0.024, +0.047] | 109.482 | 52.471 |
| hybrid | ToolRet-trained-bge-base-en-v1.5 | toolret | query | routed3 | exact | 0.383 | 0.460 | 0.538 | 0.582 | 0.627 | 0.566 | 0.661 | 0.709 | 0.239 | 0.924 | +0.034 [+0.022, +0.045] | 109.482 | 52.471 |
| bm25 | – | toolret | query | toolset | exact | 0.405 | 0.465 | 0.529 | 0.559 | 0.578 | 0.570 | 0.648 | 0.686 | 0.239 | – | +0.025 [+0.011, +0.038] | – | – |
| bm25 | – | toolret | query | toolset | duplicate | 0.406 | 0.466 | 0.529 | 0.560 | 0.578 | 0.571 | 0.648 | 0.687 | 0.239 | – | +0.025 [+0.011, +0.038] | – | – |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | expanded | query | routed3 | duplicate | 0.375 | 0.443 | 0.529 | 0.587 | 0.638 | 0.545 | 0.650 | 0.721 | 0.233 | 0.929 | +0.025 [+0.015, +0.034] | 22.713 | 9.039 |
| dense | ToolRet-trained-bge-base-en-v1.5 | toolret | query | pooled | duplicate | 0.372 | 0.443 | 0.528 | 0.590 | 0.643 | 0.548 | 0.647 | 0.726 | 0.238 | – | +0.024 [+0.009, +0.037] | 109.482 | 52.471 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | expanded | query | pooled | duplicate | 0.376 | 0.441 | 0.528 | 0.583 | 0.640 | 0.544 | 0.652 | 0.720 | 0.225 | – | +0.023 [+0.014, +0.033] | 22.713 | 9.039 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | expanded | query | routed3 | exact | 0.369 | 0.439 | 0.527 | 0.586 | 0.637 | 0.539 | 0.647 | 0.719 | 0.233 | 0.929 | +0.022 [+0.013, +0.032] | 22.713 | 9.039 |
| dense | ToolRet-trained-bge-base-en-v1.5 | toolret | query | pooled | exact | 0.369 | 0.440 | 0.526 | 0.588 | 0.642 | 0.544 | 0.644 | 0.724 | 0.238 | – | +0.022 [+0.007, +0.035] | 109.482 | 52.471 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | expanded | query | pooled | exact | 0.370 | 0.436 | 0.525 | 0.581 | 0.639 | 0.538 | 0.649 | 0.717 | 0.225 | – | +0.021 [+0.012, +0.030] | 22.713 | 9.039 |
| dense | ToolRet-trained-bge-base-en-v1.5 | toolret | query | routed3 | duplicate | 0.370 | 0.442 | 0.521 | 0.573 | 0.616 | 0.544 | 0.635 | 0.703 | 0.242 | 0.904 | +0.016 [+0.001, +0.030] | 109.482 | 52.471 |
| dense | ToolRet-trained-bge-base-en-v1.5 | toolret | query | routed3 | exact | 0.367 | 0.440 | 0.518 | 0.572 | 0.614 | 0.540 | 0.632 | 0.701 | 0.242 | 0.904 | +0.014 [-0.001, +0.028] | 109.482 | 52.471 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | toolret | query | pooled | duplicate | 0.350 | 0.425 | 0.505 | 0.560 | 0.614 | 0.521 | 0.620 | 0.688 | 0.226 | – | +0.001 [+0.000, +0.003] | 22.713 | 9.039 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | toolret | query | pooled | exact | 0.348 | 0.424 | 0.504 | 0.560 | 0.613 | 0.519 | 0.618 | 0.687 | 0.226 | – | – | 22.713 | 9.039 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | toolret | query | routed3 | duplicate | 0.352 | 0.425 | 0.502 | 0.561 | 0.607 | 0.518 | 0.612 | 0.684 | 0.234 | 0.921 | -0.002 [-0.008, +0.004] | 22.713 | 9.039 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | toolret | query | routed3 | exact | 0.349 | 0.423 | 0.501 | 0.561 | 0.606 | 0.515 | 0.611 | 0.683 | 0.234 | 0.921 | -0.003 [-0.009, +0.003] | 22.713 | 9.039 |
| dense | multi-qa-MiniLM-L6-cos-v1 | expanded | query | routed3 | duplicate | 0.334 | 0.386 | 0.475 | 0.537 | 0.594 | 0.472 | 0.581 | 0.657 | 0.215 | 0.907 | -0.030 [-0.043, -0.017] | 22.713 | 9.039 |
| dense | multi-qa-MiniLM-L6-cos-v1 | expanded | query | routed3 | exact | 0.331 | 0.384 | 0.473 | 0.535 | 0.593 | 0.470 | 0.579 | 0.654 | 0.215 | 0.907 | -0.031 [-0.045, -0.019] | 22.713 | 9.039 |
| dense | multi-qa-MiniLM-L6-cos-v1 | expanded | query | pooled | duplicate | 0.335 | 0.386 | 0.471 | 0.540 | 0.611 | 0.473 | 0.580 | 0.663 | 0.206 | – | -0.033 [-0.047, -0.021] | 22.713 | 9.039 |
| dense | multi-qa-MiniLM-L6-cos-v1 | expanded | query | pooled | exact | 0.332 | 0.384 | 0.469 | 0.538 | 0.610 | 0.471 | 0.577 | 0.661 | 0.206 | – | -0.035 [-0.049, -0.023] | 22.713 | 9.039 |
| bm25 | – | expanded | query | pooled | duplicate | 0.325 | 0.377 | 0.455 | 0.509 | 0.555 | 0.467 | 0.560 | 0.626 | 0.199 | – | -0.049 [-0.062, -0.036] | – | – |
| bm25 | – | expanded | query | pooled | exact | 0.320 | 0.372 | 0.451 | 0.506 | 0.553 | 0.460 | 0.554 | 0.621 | 0.199 | – | -0.053 [-0.067, -0.041] | – | – |
| dense | multi-qa-MiniLM-L6-cos-v1 | toolret | query | routed3 | duplicate | 0.310 | 0.363 | 0.450 | 0.509 | 0.567 | 0.441 | 0.543 | 0.613 | 0.224 | 0.891 | -0.054 [-0.066, -0.042] | 22.713 | 9.039 |
| dense | multi-qa-MiniLM-L6-cos-v1 | toolret | query | routed3 | exact | 0.308 | 0.362 | 0.449 | 0.508 | 0.566 | 0.439 | 0.541 | 0.611 | 0.224 | 0.891 | -0.055 [-0.068, -0.044] | 22.713 | 9.039 |
| bm25 | – | expanded | query | routed3 | duplicate | 0.324 | 0.376 | 0.448 | 0.493 | 0.526 | 0.465 | 0.546 | 0.601 | 0.209 | 0.896 | -0.056 [-0.070, -0.043] | – | – |
| dense | multi-qa-MiniLM-L6-cos-v1 | toolret | query | pooled | duplicate | 0.311 | 0.364 | 0.448 | 0.516 | 0.590 | 0.442 | 0.542 | 0.624 | 0.219 | – | -0.056 [-0.068, -0.045] | 22.713 | 9.039 |
| dense | multi-qa-MiniLM-L6-cos-v1 | toolret | query | pooled | exact | 0.308 | 0.363 | 0.446 | 0.514 | 0.589 | 0.441 | 0.540 | 0.622 | 0.219 | – | -0.058 [-0.070, -0.047] | 22.713 | 9.039 |
| bm25 | – | expanded | query | routed3 | exact | 0.319 | 0.371 | 0.444 | 0.490 | 0.524 | 0.458 | 0.540 | 0.597 | 0.209 | 0.896 | -0.060 [-0.074, -0.047] | – | – |
| bm25 | – | toolret | query | pooled | exact | 0.298 | 0.356 | 0.420 | 0.472 | 0.511 | 0.439 | 0.515 | 0.579 | 0.187 | – | -0.085 [-0.097, -0.072] | – | – |
| bm25 | – | toolret | query | pooled | duplicate | 0.298 | 0.356 | 0.420 | 0.472 | 0.512 | 0.439 | 0.515 | 0.579 | 0.187 | – | -0.085 [-0.097, -0.072] | – | – |
| bm25 | – | toolret | query | routed3 | exact | 0.296 | 0.348 | 0.413 | 0.454 | 0.488 | 0.427 | 0.504 | 0.551 | 0.191 | 0.881 | -0.091 [-0.105, -0.078] | – | – |
| bm25 | – | toolret | query | routed3 | duplicate | 0.297 | 0.348 | 0.413 | 0.454 | 0.488 | 0.427 | 0.504 | 0.551 | 0.191 | 0.881 | -0.091 [-0.105, -0.078] | – | – |

## Recall@50 by catalog

| method | encoder | docs | query | scope | match | code | customized |
|---|---|---|---|---|---|---|---|
| dense | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | expanded | query | toolset | duplicate | 0.830 | 0.622 |
| dense | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | expanded | query | toolset | exact | 0.828 | 0.622 |
| hybrid | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | expanded | query | toolset | duplicate | 0.828 | 0.619 |
| hybrid | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | expanded | query | toolset | exact | 0.826 | 0.619 |
| hybrid | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | toolret | query | toolset | duplicate | 0.815 | 0.618 |
| dense | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | toolret | query | toolset | duplicate | 0.808 | 0.629 |
| dense | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | toolret | query | toolset | exact | 0.805 | 0.629 |
| hybrid | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | toolret | query | toolset | exact | 0.811 | 0.618 |
| dense | Tool-Embed-0.6B | expanded | query | toolset | duplicate | 0.810 | 0.619 |
| dense | Tool-Embed-0.6B | expanded | query | toolset | exact | 0.806 | 0.619 |
| hybrid | Tool-Embed-0.6B | expanded | query | toolset | duplicate | 0.794 | 0.618 |
| hybrid | Tool-Embed-0.6B | toolret | query | toolset | duplicate | 0.794 | 0.614 |
| hybrid | Tool-Embed-0.6B | toolret | query | toolset | exact | 0.792 | 0.614 |
| dense | Tool-Embed-0.6B | toolret | query | toolset | duplicate | 0.788 | 0.620 |
| hybrid | Tool-Embed-0.6B | expanded | query | toolset | exact | 0.788 | 0.618 |
| dense | Tool-Embed-0.6B | toolret | query | toolset | exact | 0.786 | 0.620 |
| dense | ToolRet-trained-bge-base-en-v1.5 | expanded | query | toolset | duplicate | 0.759 | 0.573 |
| dense | ToolRet-trained-bge-base-en-v1.5 | expanded | query | toolset | exact | 0.757 | 0.573 |
| hybrid | ToolRet-trained-bge-base-en-v1.5 | expanded | query | toolset | duplicate | 0.748 | 0.571 |
| hybrid | ToolRet-trained-bge-base-en-v1.5 | expanded | query | toolset | exact | 0.744 | 0.571 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | expanded | query | toolset | duplicate | 0.718 | 0.598 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | expanded | query | toolset | exact | 0.712 | 0.598 |
| dense | ToolRet-trained-bge-base-en-v1.5 | toolret | query | toolset | duplicate | 0.728 | 0.566 |
| dense | ToolRet-trained-bge-base-en-v1.5 | toolret | query | toolset | exact | 0.726 | 0.566 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | toolret | query | toolset | duplicate | 0.709 | 0.594 |
| hybrid | ToolRet-trained-bge-base-en-v1.5 | toolret | query | toolset | duplicate | 0.724 | 0.563 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | toolret | query | toolset | exact | 0.707 | 0.594 |
| hybrid | ToolRet-trained-bge-base-en-v1.5 | toolret | query | toolset | exact | 0.723 | 0.563 |
| dense | multi-qa-MiniLM-L6-cos-v1 | expanded | query | toolset | duplicate | 0.696 | 0.600 |
| dense | multi-qa-MiniLM-L6-cos-v1 | expanded | query | toolset | exact | 0.695 | 0.600 |
| dense | multi-qa-MiniLM-L6-cos-v1 | toolret | query | toolset | duplicate | 0.692 | 0.601 |
| dense | multi-qa-MiniLM-L6-cos-v1 | toolret | query | toolset | exact | 0.689 | 0.601 |
| dense | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | expanded | query | routed3 | duplicate | 0.680 | 0.491 |
| dense | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | expanded | query | routed3 | exact | 0.678 | 0.491 |
| dense | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | expanded | query | pooled | duplicate | 0.683 | 0.480 |
| dense | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | expanded | query | pooled | exact | 0.680 | 0.480 |
| hybrid | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | toolret | query | pooled | duplicate | 0.677 | 0.483 |
| hybrid | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | toolret | query | pooled | exact | 0.672 | 0.483 |
| hybrid | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | toolret | query | routed3 | duplicate | 0.676 | 0.476 |
| hybrid | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | toolret | query | routed3 | exact | 0.672 | 0.476 |
| dense | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | toolret | query | pooled | duplicate | 0.670 | 0.467 |
| dense | Tool-Embed-0.6B | expanded | query | routed3 | duplicate | 0.662 | 0.481 |
| dense | Tool-Embed-0.6B | expanded | query | pooled | duplicate | 0.663 | 0.479 |
| hybrid | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | expanded | query | routed3 | duplicate | 0.655 | 0.490 |
| dense | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | toolret | query | pooled | exact | 0.667 | 0.467 |
| hybrid | Tool-Embed-0.6B | expanded | query | routed3 | duplicate | 0.662 | 0.475 |
| hybrid | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | expanded | query | pooled | duplicate | 0.655 | 0.486 |
| hybrid | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | expanded | query | routed3 | exact | 0.652 | 0.490 |
| dense | Tool-Embed-0.6B | expanded | query | routed3 | exact | 0.657 | 0.481 |
| dense | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | toolret | query | routed3 | duplicate | 0.668 | 0.461 |
| dense | Tool-Embed-0.6B | expanded | query | pooled | exact | 0.658 | 0.479 |
| hybrid | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | expanded | query | pooled | exact | 0.652 | 0.486 |
| dense | Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | toolret | query | routed3 | exact | 0.665 | 0.461 |
| hybrid | Tool-Embed-0.6B | expanded | query | routed3 | exact | 0.657 | 0.475 |
| hybrid | Tool-Embed-0.6B | toolret | query | routed3 | duplicate | 0.654 | 0.477 |
| dense | Tool-Embed-0.6B | toolret | query | routed3 | duplicate | 0.646 | 0.490 |
| hybrid | Tool-Embed-0.6B | expanded | query | pooled | duplicate | 0.666 | 0.453 |
| hybrid | Tool-Embed-0.6B | toolret | query | routed3 | exact | 0.652 | 0.477 |
| dense | Tool-Embed-0.6B | toolret | query | routed3 | exact | 0.644 | 0.490 |
| dense | Tool-Embed-0.6B | toolret | query | pooled | duplicate | 0.646 | 0.482 |
| dense | Tool-Embed-0.6B | toolret | query | pooled | exact | 0.644 | 0.482 |
| hybrid | Tool-Embed-0.6B | toolret | query | pooled | duplicate | 0.656 | 0.459 |
| hybrid | Tool-Embed-0.6B | expanded | query | pooled | exact | 0.660 | 0.453 |
| hybrid | Tool-Embed-0.6B | toolret | query | pooled | exact | 0.655 | 0.459 |
| bm25 | – | expanded | query | toolset | duplicate | 0.631 | 0.452 |
| bm25 | – | expanded | query | toolset | exact | 0.623 | 0.452 |
| dense | ToolRet-trained-bge-base-en-v1.5 | expanded | query | pooled | duplicate | 0.626 | 0.447 |
| hybrid | ToolRet-trained-bge-base-en-v1.5 | expanded | query | routed3 | duplicate | 0.627 | 0.446 |
| hybrid | ToolRet-trained-bge-base-en-v1.5 | expanded | query | pooled | duplicate | 0.625 | 0.447 |
| dense | ToolRet-trained-bge-base-en-v1.5 | expanded | query | pooled | exact | 0.624 | 0.447 |
| hybrid | ToolRet-trained-bge-base-en-v1.5 | expanded | query | routed3 | exact | 0.623 | 0.446 |
| hybrid | ToolRet-trained-bge-base-en-v1.5 | expanded | query | pooled | exact | 0.620 | 0.447 |
| dense | ToolRet-trained-bge-base-en-v1.5 | expanded | query | routed3 | duplicate | 0.621 | 0.443 |
| dense | ToolRet-trained-bge-base-en-v1.5 | expanded | query | routed3 | exact | 0.619 | 0.443 |
| hybrid | ToolRet-trained-bge-base-en-v1.5 | toolret | query | pooled | duplicate | 0.604 | 0.440 |
| hybrid | ToolRet-trained-bge-base-en-v1.5 | toolret | query | pooled | exact | 0.603 | 0.440 |
| hybrid | ToolRet-trained-bge-base-en-v1.5 | toolret | query | routed3 | duplicate | 0.599 | 0.433 |
| hybrid | ToolRet-trained-bge-base-en-v1.5 | toolret | query | routed3 | exact | 0.597 | 0.433 |
| bm25 | – | toolret | query | toolset | exact | 0.588 | 0.424 |
| bm25 | – | toolret | query | toolset | duplicate | 0.588 | 0.424 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | expanded | query | routed3 | duplicate | 0.589 | 0.421 |
| dense | ToolRet-trained-bge-base-en-v1.5 | toolret | query | pooled | duplicate | 0.585 | 0.427 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | expanded | query | pooled | duplicate | 0.595 | 0.408 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | expanded | query | routed3 | exact | 0.586 | 0.421 |
| dense | ToolRet-trained-bge-base-en-v1.5 | toolret | query | pooled | exact | 0.581 | 0.427 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | expanded | query | pooled | exact | 0.591 | 0.408 |
| dense | ToolRet-trained-bge-base-en-v1.5 | toolret | query | routed3 | duplicate | 0.581 | 0.413 |
| dense | ToolRet-trained-bge-base-en-v1.5 | toolret | query | routed3 | exact | 0.577 | 0.413 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | toolret | query | pooled | duplicate | 0.561 | 0.406 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | toolret | query | pooled | exact | 0.559 | 0.406 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | toolret | query | routed3 | duplicate | 0.557 | 0.404 |
| hybrid | multi-qa-MiniLM-L6-cos-v1 | toolret | query | routed3 | exact | 0.556 | 0.404 |
| dense | multi-qa-MiniLM-L6-cos-v1 | expanded | query | routed3 | duplicate | 0.531 | 0.374 |
| dense | multi-qa-MiniLM-L6-cos-v1 | expanded | query | routed3 | exact | 0.528 | 0.374 |
| dense | multi-qa-MiniLM-L6-cos-v1 | expanded | query | pooled | duplicate | 0.533 | 0.360 |
| dense | multi-qa-MiniLM-L6-cos-v1 | expanded | query | pooled | exact | 0.530 | 0.360 |
| bm25 | – | expanded | query | pooled | duplicate | 0.515 | 0.348 |
| bm25 | – | expanded | query | pooled | exact | 0.508 | 0.348 |
| dense | multi-qa-MiniLM-L6-cos-v1 | toolret | query | routed3 | duplicate | 0.491 | 0.378 |
| dense | multi-qa-MiniLM-L6-cos-v1 | toolret | query | routed3 | exact | 0.489 | 0.378 |
| bm25 | – | expanded | query | routed3 | duplicate | 0.506 | 0.344 |
| dense | multi-qa-MiniLM-L6-cos-v1 | toolret | query | pooled | duplicate | 0.492 | 0.370 |
| dense | multi-qa-MiniLM-L6-cos-v1 | toolret | query | pooled | exact | 0.489 | 0.370 |
| bm25 | – | expanded | query | routed3 | exact | 0.500 | 0.344 |
| bm25 | – | toolret | query | pooled | exact | 0.478 | 0.316 |
| bm25 | – | toolret | query | pooled | duplicate | 0.478 | 0.316 |
| bm25 | – | toolret | query | routed3 | exact | 0.471 | 0.311 |
| bm25 | – | toolret | query | routed3 | duplicate | 0.471 | 0.311 |
