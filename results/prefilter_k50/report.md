# Prefilter at small K

1936 single-tool steps from ToolRet code, customized. Each step's own toolset is ranked (the `toolset` scope); recall@K = the gold tool is in the top K. Base = RRF ensemble of the encoders over Tool-REX expanded docs (the section 7 prefilter, with ranks taken within the toolset). Usage kNN = query-to-query kNN over past requests from the other 5 cross-fitting folds of the same toolset (same-text requests left out); the gold tool had been used before for 39% of steps. Gorilla train requests = APIBench's training instructions (8131 requests for 907 Hugging Face models; ToolRet's gorilla-huggingface requests come from its eval split; 281 requests within 0.95 cosine of an evaluated request dropped), scored per tool by the best match (`kNN`) or the mean embedding (`centroid`, Tool2Vec) and fused with RRF. `gated` = the usage weight and similarity floor chosen on the other folds. ΔR@50 = paired bootstrap against the base.

| method | R@10 | R@20 | R@30 | R@50 | R@100 | R@200 | K for 95% | ΔR@50 vs base [95% CI] |
|---|---|---|---|---|---|---|---|---|
| base + ToolRet-trained-e5-large-v2 + Gorilla train requests (kNN) + usage kNN (gated) | 0.763 | 0.853 | 0.895 | 0.934 | 0.965 | 0.988 | 67 | +0.038 [+0.029, +0.048] |
| base + ToolRet-trained-e5-large-v2 + Gorilla train requests (centroid) | 0.730 | 0.824 | 0.875 | 0.931 | 0.965 | 0.988 | 69 | +0.035 [+0.026, +0.044] |
| base + ToolRet-trained-e5-large-v2 + Gorilla train requests (kNN) | 0.727 | 0.821 | 0.871 | 0.929 | 0.964 | 0.988 | 71 | +0.034 [+0.025, +0.043] |
| base: Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 + Gorilla train requests (kNN) + usage kNN (gated) | 0.727 | 0.841 | 0.887 | 0.925 | 0.965 | 0.987 | 69 | +0.029 [+0.022, +0.038] |
| base: Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 + Gorilla train requests (centroid) | 0.717 | 0.814 | 0.866 | 0.921 | 0.966 | 0.988 | 70 | +0.025 [+0.019, +0.033] |
| base + ToolRet-trained-e5-large-v2 + usage kNN (gated) | 0.746 | 0.830 | 0.874 | 0.918 | 0.954 | 0.979 | 92 | +0.023 [+0.014, +0.032] |
| base: Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 + Gorilla train requests (kNN) | 0.713 | 0.812 | 0.861 | 0.915 | 0.965 | 0.986 | 73 | +0.020 [+0.013, +0.026] |
| base + ToolRet-trained-e5-large-v2 | 0.715 | 0.802 | 0.853 | 0.913 | 0.952 | 0.978 | 95 | +0.018 [+0.010, +0.026] |
| base: Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 + usage kNN (gated) | 0.700 | 0.824 | 0.869 | 0.912 | 0.947 | 0.974 | 106 | +0.016 [+0.009, +0.024] |
| base + ToolRet-trained-bge-large-en-v1.5 + ToolRet-trained-e5-large-v2 | 0.709 | 0.806 | 0.851 | 0.908 | 0.956 | 0.978 | 91 | +0.012 [+0.004, +0.021] |
| base + ToolRet-trained-bge-large-en-v1.5 | 0.702 | 0.796 | 0.843 | 0.901 | 0.956 | 0.977 | 89 | +0.006 [-0.002, +0.014] |
| base: Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | 0.697 | 0.794 | 0.841 | 0.896 | 0.948 | 0.973 | 110 | – |
| base + ToolRet-trained-e5-large-v2 + usage kNN (plain RRF) | 0.591 | 0.794 | 0.848 | 0.894 | 0.939 | 0.971 | 125 | -0.002 [-0.012, +0.010] |
| ToolRet-trained-e5-large-v2 | 0.661 | 0.766 | 0.818 | 0.885 | 0.937 | 0.970 | 122 | -0.010 [-0.022, +0.003] |
| Tool-Embed-0.6B | 0.655 | 0.749 | 0.813 | 0.876 | 0.919 | 0.958 | 166 | -0.020 [-0.031, -0.010] |
| ToolRet-trained-bge-large-en-v1.5 | 0.668 | 0.752 | 0.808 | 0.863 | 0.935 | 0.976 | 122 | -0.033 [-0.046, -0.019] |
| base: Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 + usage kNN (plain RRF) | 0.468 | 0.683 | 0.807 | 0.863 | 0.925 | 0.962 | 159 | -0.033 [-0.046, -0.021] |
| ToolRet-trained-bge-base-en-v1.5 | 0.651 | 0.762 | 0.806 | 0.856 | 0.926 | 0.971 | 142 | -0.040 [-0.052, -0.028] |
| usage kNN only | 0.353 | 0.397 | 0.402 | 0.422 | 0.471 | 0.686 | 335 | -0.474 [-0.497, -0.450] |

## Recall@50 by dataset

| dataset | n | base + ToolRet-trained-e5-large-v2 + Gorilla train requests (kNN) + usage kNN (gated) | base + ToolRet-trained-e5-large-v2 + Gorilla train requests (centroid) | base + ToolRet-trained-e5-large-v2 + Gorilla train requests (kNN) | base: Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 + Gorilla train requests (kNN) + usage kNN (gated) | base: Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 + Gorilla train requests (centroid) | base + ToolRet-trained-e5-large-v2 + usage kNN (gated) |
|---|---|---|---|---|---|---|---|
| appbench | 11 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| craft-math-algebra | 280 | 0.989 | 0.989 | 0.989 | 0.989 | 0.989 | 0.989 |
| craft-tabmwp | 174 | 0.845 | 0.845 | 0.845 | 0.851 | 0.851 | 0.845 |
| craft-vqa | 200 | 0.805 | 0.805 | 0.805 | 0.740 | 0.745 | 0.805 |
| gorilla-huggingface | 500 | 0.900 | 0.902 | 0.896 | 0.894 | 0.918 | 0.840 |
| gorilla-pytorch | 43 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| gorilla-tensor | 55 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| gpt4tools | 32 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| gta | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| metatool | 179 | 0.989 | 0.989 | 0.989 | 0.972 | 0.972 | 0.989 |
| taskbench-huggingface | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| taskbench-multimedia | 16 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| toolalpaca | 94 | 0.968 | 0.968 | 0.968 | 0.979 | 0.968 | 0.968 |
| toolbench-sam | 191 | 1.000 | 0.963 | 0.963 | 0.995 | 0.901 | 1.000 |
| toolink | 149 | 0.973 | 0.973 | 0.973 | 0.980 | 0.966 | 0.973 |

## Adaptive K

Candidates per request calibrated for 95% coverage on the other folds (split conformal, cross-fitted). `fixed K`: one K for all; `K per toolset`: one K per toolset (Mondrian conformal); `score margin`: keep every tool whose score is within a calibrated margin of the request's best tool, globally or per toolset. K is capped at the toolset size.

| ranking | rule | coverage | mean K | median K | 90th pct K |
|---|---|---|---|---|---|
| base: Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | fixed K | 0.950 | 104.832 | 110.000 | 115.000 |
| base: Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | K per toolset | 0.955 | 120.229 | 62.000 | 251.000 |
| base: Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | score margin | 0.950 | 95.279 | 95.000 | 120.000 |
| base: Tool-Embed-0.6B + ToolRet-trained-bge-base-en-v1.5 | score margin per toolset | 0.954 | 120.771 | 62.000 | 265.500 |
| base + ToolRet-trained-e5-large-v2 + Gorilla train requests (kNN) + usage kNN (gated) | fixed K | 0.951 | 66.122 | 69.000 | 74.000 |
| base + ToolRet-trained-e5-large-v2 + Gorilla train requests (kNN) + usage kNN (gated) | K per toolset | 0.955 | 74.828 | 40.000 | 242.000 |
| base + ToolRet-trained-e5-large-v2 + Gorilla train requests (kNN) + usage kNN (gated) | score margin | 0.950 | 79.053 | 57.000 | 145.000 |
| base + ToolRet-trained-e5-large-v2 + Gorilla train requests (kNN) + usage kNN (gated) | score margin per toolset | 0.955 | 77.702 | 37.000 | 224.000 |
| mean cosine of the base encoders | fixed K | 0.950 | 99.717 | 105.000 | 111.000 |
| mean cosine of the base encoders | K per toolset | 0.955 | 119.170 | 62.000 | 267.000 |
| mean cosine of the base encoders | score margin | 0.950 | 147.162 | 58.000 | 442.500 |
| mean cosine of the base encoders | score margin per toolset | 0.954 | 121.010 | 43.000 | 393.000 |
