# Results so far (12 Oct 2026)

All runs use ToolRet (7,961 steps, 44,453 tools). The Stage-1 prefilter is BM25 + `multi-qa-MiniLM-L6-cos-v1` fused with RRF, K = 50 (section 5 varies K; sections 6–8 test stronger prefilters). The decision model is Inception Mercury Decide on OpenRouter (free tier). Detailed tables and figures are in each `results/**/report.md`.

## 1. Tests

`pytest`: 32 tests pass. They check that:

- **Coverage guarantee (synthetic data):** split conformal hits 1 − α for LAC and APS.
- **Risk control:** conformal risk control keeps the false-negative rate ≤ α, and its exact threshold matches brute force.
- **Prefilter misses:** they push q̂ to ∞, and coverage is then capped at recall@K.
- **Online ACI:** it restores coverage after a score shift.
- **API clients:** OpenRouter, OpenAI Decisions and Chat Completions work against mocked HTTP endpoints, including retry on 429 and caching.
- **Prefilter scopes:** the agent's own toolset is ranked first, and routing follows the best-matching toolset.
- **LangChain middleware:** the agent first sees only `find_tool`, then only the conformal set. Unexposed tools are blocked, governance blocks destructive tools, and an empty set falls back to top-1.

## 2. Phase 1 on the full dataset, local score sources (RQ1)

6,369 eval steps (plus 1,592 dev steps for calibration fitting), 100 random 50/50 calibration/test splits. `results/minilm/phase1/report.md`

**Single-tool steps whose gold tool survived the prefilter (n = 2,165), LAC:**

| score source | top-1 acc. | coverage at α = 0.1 (mean / 5th pct) | avg set size, α = 0.05 / 0.1 / 0.2 | coverage if capped at 5 |
|---|---|---|---|---|
| BM25 | 0.21 | 0.900 / 0.876 | 38.8 / 29.2 / 18.9 | 0.57 |
| dense (MiniLM) | 0.15 | 0.903 / 0.885 | 38.1 / 33.3 / 25.2 | 0.49 |
| hybrid (RRF) | 0.22 | 0.901 / 0.876 | 30.9 / 21.0 / 10.1 | 0.68 |
| cross-encoder (choice) | 0.26 | 0.903 / 0.881 | 23.8 / 16.5 / 8.2 | 0.70 |
| cross-encoder (yes/no, Platt) | 0.26 | 0.901 / 0.883 | 36.8 / 28.1 / 18.6 | 0.69 |

- **The guarantee holds at scale.** Mean coverage is within 0.005 of 1 − α for every source and α. The spread across splits matches the Beta-distribution theory (theoretical 5th percentile 0.885, observed 0.876–0.885). See `coverage_distribution.png`.
- **Better scores give smaller sets.** The cross-encoder's sets are 16.8 tools smaller than dense at the same coverage (95% CI [−17.3, −16.2]). The same model scored as independent yes/no probabilities gives sets 70% larger than as one choice distribution, so the choice mode (Mode A) is the more efficient way to ask.
- **Capping at 5 tools breaks the guarantee badly with local scorers** (coverage 0.49–0.70). It is only safe with a sharp scorer like the decision model below.
- **Coverage is marginal, not per catalog.** At α = 0.1 the web catalog is over-covered (0.90–0.97) and the code catalog under-covered (0.80–0.88). Per-catalog (Mondrian) calibration would fix this.
- **Multi-tool steps (n = 799), conformal risk control:** the false-negative rate is held at 0.097–0.100 for α = 0.1. Full coverage of every needed tool is about 0.80, as expected, because risk control bounds missed tools, not whole steps.

## 3. The prefilter is the bottleneck (design §8.4 confirmed)

| | K = 20 | K = 50 | K = 100 |
|---|---|---|---|
| dev recall (all gold tools in top-K) | 0.37 | 0.46 | 0.51 |

End to end, q̂ = ∞ for every α ≤ 0.2 and every score source, so the set is all 50 tools. Coverage is then capped at recall: **0.62** for single-tool steps and 0.28 for multi-tool steps. No Stage-2 scorer can fix this, so a stronger prefilter is the main lever; section 6 reviews and tests the options.

## 4. Decision model smoke test (H1), 47 Mercury Decide calls

59 single-tool steps (12 dev + 47 eval); Stage-2 view n = 28. `results/mercury_smoke/phase1/report.md`

| score source | top-1 acc. | coverage at α = 0.1 | avg tools shown (of 50) | coverage if capped at 5 |
|---|---|---|---|---|
| cross-encoder | 0.28 | 0.94 | 28.2 | 0.64 |
| **Mercury Decide** | **0.45** | **0.94** | **2.3** | **0.94** |

- **At matched coverage, Mercury's sets are about 12× smaller than the cross-encoder's.** Mercury against dense: −40.8 tools, 95% CI [−42.3, −39.3]. This is the direction H1 predicts. n = 28, so treat it as a strong signal, not a result.
- **Mercury is overconfident:** ECE 0.26, with confidence near 1.0 but accuracy near 0.7. Conformal calibration absorbs this; a raw confidence threshold would not.
- **Usage:** median API latency 1.1 s, about 2.7k input tokens per call, $0.

## 5. Prefilter K sweep: dense and Mercury Decide (variable K)

`results/k_sweep/report.md` (`scripts/run_k_sweep.py`). These are the same 59 single-tool steps as section 4 (12 dev + 47 eval). Only K varies; the rest of the pipeline is unchanged. Mercury Decide was run at K = 100 (47 calls). Its K = 50 row is the section 4 run on the same steps.

**Prefilter results** (recall = right tool among the K candidates):

| K | 10 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|
| eval recall (n = 47) | 0.43 | 0.49 | 0.55 | 0.60 | 0.68 |
| dev recall (n = 12) | 0.17 | 0.33 | 0.33 | 0.50 | 0.67 |

**Final results** (single-tool steps; hit@k = right tool among the top k shown; LAC at α = 0.1):

| score source | K | e2e hit@1 | e2e hit@5 | Stage-2 n | Stage-2 hit@1 | Stage-2 hit@5 | Stage-2 coverage | Stage-2 avg set size | e2e coverage |
|---|---|---|---|---|---|---|---|---|---|
| dense | 10 | 0.17 | 0.32 | 20 | 0.40 | 0.75 | 0.90 | 8.7 | 0.43 |
| dense | 20 | 0.17 | 0.30 | 23 | 0.35 | 0.61 | 0.92 | 16.2 | 0.49 |
| dense | 30 | 0.17 | 0.26 | 26 | 0.31 | 0.46 | 0.92 | 26.5 | 0.55 |
| dense | 50 | 0.17 | 0.26 | 28 | 0.29 | 0.43 | 0.93 | 43.1 | 0.59 |
| dense | 100 | 0.17 | 0.26 | 32 | 0.25 | 0.38 | 0.94 | 83.7 | 0.67 |
| Mercury Decide | 50 (section 4) | 0.45 | 0.57 | 28 | 0.75 | 0.96 | 0.94 | 2.3 | 0.59 |
| **Mercury Decide** | **100** | **0.47** | **0.57** | **32** | **0.69** | **0.84** | **0.95** | **5.0** | **0.67** |

- **Recall rises slowly with K:** 0.43 at K = 10, 0.60 at K = 50, 0.68 at K = 100. No K reaches 0.90, so end to end the threshold is ∞ at α = 0.1 for every K and source. The agent then sees all K tools, and end-to-end coverage equals recall.
- **Going from K = 50 to 100 barely helps Mercury end to end.** 4 more steps reach Stage 2 (28 → 32), but top-1 rises by only one step (21 → 22 of 47), and hit@5 is unchanged at 0.57.
- **Mercury holds up with 100 candidates on the same steps.** On the 28 steps that reached Stage 2 at K = 50, its hit@1 at K = 100 is unchanged (0.75), and hit@5 drops from 0.96 to 0.89. The 4 steps that K = 100 adds are harder (hit@1 1/4). That is why the Stage-2 hit@1 falls to 0.69, and its conformal set at α = 0.1 grows from 2.3 to 5.0 tools. Coverage when capped at 5 tools falls from 0.94 to 0.85.
- **Dense sets grow almost linearly with K** (8.7 → 83.7 tools at α = 0.1). Dense cannot separate the right tool from the added candidates; its Stage-2 hit@1 falls from 0.40 to 0.25.
- **Usage at K = 100:** about 5.3k input tokens per call (2.7k at K = 50), median latency 0.9 s, 0 refusals.
- n = 47 (32 at Stage 2), so treat these as signals, not results. A larger K alone does not fix the prefilter; a stronger Stage-1 retriever is still the main lever (section 3).

## 6. Better prefilters: literature review and tests

The prefilter caps end-to-end coverage (sections 3 and 5), so this section looks for a stronger Stage 1. `scripts/run_prefilter_bench.py` measures recall@K (all gold tools in the top K) with no API calls. `results/prefilter/*/report.md`.

**What the literature suggests**

| approach | idea | reported result | tested here |
|---|---|---|---|
| Tool-trained embedders: ToolRet-train ([Shi et al., ACL Findings 2025](https://arxiv.org/abs/2503.01763)) | fine-tune retrievers on 200k tool-retrieval tasks | bge-base NDCG@10 on ToolRet 22.1 → 29.4 (their Table 14) | `ToolRet-trained-bge-base-en-v1.5` |
| Document expansion + tool-trained embedder: Tool-DE / Tool-Embed ([Lu et al., 2025](https://arxiv.org/abs/2510.22670)) | LLM-written tool profiles (function, when to use, tags); embedder trained on them | Tool-Embed-0.6B +4.97 NDCG@10 over its base, Qwen3-Embedding-0.6B (43.13); Tool-Embed-4B best on ToolRet and Tool-DE | expanded docs (Tool-REX); `Tool-Embed-0.6B` |
| Query-side intent: Re-Invoke ([Chen et al., EMNLP Findings 2024](https://arxiv.org/abs/2408.01875)); MCP-Zero ([Fei et al., 2025](https://arxiv.org/abs/2506.01056)) | an LLM extracts the tool intent, or the agent writes a structured tool request | Re-Invoke +20% nDCG@5 single-tool, +39% multi-tool (ToolE); MCP-Zero 98% fewer tokens on APIBank | upper bound only (ToolRet instruction) |
| Usage-based tool embeddings: Tool2Vec ([Moon et al., 2024](https://arxiv.org/abs/2409.02141)) | embed tools from example queries, not descriptions | up to +27.28 Recall@K on ToolBench | no: needs usage queries per tool |
| Graph / completeness: COLT ([Qu et al., CIKM 2024](https://arxiv.org/abs/2405.16089)) | collaborative learning over query–scene–tool graphs, for multi-tool sets | gains on ToolLens and open benchmarks | no: needs training |
| Reranking the candidates: Tool-Rank ([Lu et al., 2025](https://arxiv.org/abs/2510.22670)) | LLM reranker over the top 100 | +4.21 NDCG@10 over Tool-Embed-4B | no: 4B on CPU; Mercury already reranks at Stage 2 |
| General embedders (suggested): LFM2.5-Embedding-350M, Nemotron-3-Embed-1B | – | – | both, run locally |

**Test 1: full ToolRet** (7,961 steps, 44,453 tools; `results/prefilter/full/report.md`). BM25 and MiniLM, original vs expanded tool docs. Tool texts are cut at 600 characters here, so the baseline differs slightly from sections 2–5.

| prefilter | docs | R@10 | R@50 | R@100 | R@50 single-tool | R@50 multi-tool |
|---|---|---|---|---|---|---|
| BM25 | ToolRet | 0.28 | 0.404 | 0.46 | 0.55 | 0.23 |
| BM25 | expanded | 0.32 | 0.469 | 0.53 | 0.59 | 0.32 |
| BM25 + MiniLM, RRF (current) | ToolRet | 0.34 | 0.493 | 0.55 | 0.65 | 0.30 |
| BM25 + MiniLM, RRF | expanded | **0.37** | **0.528** | **0.59** | **0.67** | **0.35** |
| *oracle:* BM25 + MiniLM + ToolRet instruction | ToolRet | 0.44 | 0.599 | 0.66 | 0.79 | 0.37 |
| *oracle:* BM25 + MiniLM + ToolRet instruction | expanded | 0.47 | 0.635 | 0.70 | 0.81 | 0.43 |

**Test 2: tool-trained and general embedders** on the code + customized catalogs (2,731 steps, 7,161 tools; `results/prefilter/code_customized/report.md`). These two catalogs were chosen because the tool-trained models' training data does not cover them: Tool-Embed's training queries share a 12-word span with 0% of code and 0.6% of customized steps, but with 10% of web steps (mostly ToolBench and APIGen), and ToolRet-train is built from ToolACE, ToolBench and APIGen, which are all web. Every model runs locally in fp32 on 4 CPU cores. Latency is the time to encode one query on that CPU.

| prefilter | docs | R@10 | R@50 | R@100 | R@50 single-tool | ΔR@50 vs current [95% CI] | params | query ms (CPU) |
|---|---|---|---|---|---|---|---|---|
| BM25 + MiniLM, RRF (current) | ToolRet | 0.35 | 0.504 | 0.56 | 0.62 | – | 23M | 9 |
| BM25 + MiniLM, RRF | expanded | 0.37 | 0.525 | 0.58 | 0.65 | +0.021 [+0.012, +0.030] | 23M | 9 |
| BM25 + LFM2.5-Embedding-350M, RRF | ToolRet | 0.36 | 0.514 | 0.57 | 0.63 | +0.010 [−0.002, +0.022] | 354M | 158 |
| BM25 + LFM2.5-Embedding-350M, RRF | expanded | 0.38 | 0.530 | 0.59 | 0.65 | +0.026 [+0.013, +0.039] | 354M | 158 |
| BM25 + ToolRet-trained bge-base, RRF | ToolRet | 0.38 | 0.544 | 0.59 | 0.67 | +0.040 [+0.029, +0.051] | 109M | 57 |
| ToolRet-trained bge-base (dense) | expanded | 0.39 | 0.560 | 0.62 | 0.69 | +0.056 [+0.042, +0.070] | 109M | 57 |
| Tool-Embed-0.6B (dense) | ToolRet | **0.42** | 0.585 | 0.64 | 0.67 | +0.081 [+0.064, +0.098] | 596M | 355 |
| **Tool-Embed-0.6B (dense)** | **expanded** | 0.40 | **0.593** | **0.65** | **0.70** | **+0.089 [+0.072, +0.106]** | 596M | 355 |

On the customized catalog alone (982 steps; `results/prefilter/customized/report.md`), Nemotron-3-Embed-1B reaches R@50 0.437 with BM25 (+0.031 [+0.014, +0.048] over the current 0.406), below LFM2.5 (0.451) and Tool-Embed (0.482, dense), at 488 ms per query.

- **A tool-trained embedder is the largest gain we can deploy.** Tool-Embed-0.6B raises R@50 from 0.504 to 0.593 and single-tool R@50 from 0.62 to 0.70, as the Tool-DE paper predicts. Fusing it with BM25 adds nothing: dense alone is as good or better.
- **The most efficient upgrade is the ToolRet-trained bge-base with expanded docs.** It gets about 60% of Tool-Embed's gain (+0.056) at a sixth of the CPU cost (57 ms per query, 109M parameters).
- **Document expansion costs nothing at query time and always helps.** The profiles are written once per tool, offline. They add +0.036 [+0.030, +0.042] R@50 to the current prefilter on all of ToolRet and +0.065 to BM25, and they stack with the tool-trained models (+0.014 for bge, +0.008 for Tool-Embed).
- **Model size alone does not help.** The two general embedders gain at most +0.045 on customized and +0.010 (not significant) on code + customized. The 1.1B Nemotron is the slowest and is not better than the 350M LFM2.5. Of the two, LFM2.5 is the better choice. Training on tool retrieval matters more than size, as ToolRet reports.
- **Agent-written tool requests are the largest remaining lever.** Adding ToolRet's instruction to the query lifts R@50 by +0.107 [+0.099, +0.115], and by +0.142 with expanded docs. This is an upper bound, because GPT-4o wrote those instructions with the gold tools in view. A deployable version is an agent that describes the tool it needs (MCP-Zero) or an LLM that extracts the intent (Re-Invoke). The `find_tool` middleware already takes the agent's own description, so this is the next test.
- **Still far from 0.90.** The best R@100 is 0.65, so end to end the conformal threshold stays at ∞ for α = 0.1 (section 3). Combining a tool-trained embedder, expanded docs and agent-written requests is the path forward. A larger Tool-Embed-4B on a GPU should push further.
- Notes: LFM2.5-Embedding needs its own bidirectional-attention code from the model repo (`--trust-remote-code`). Without it, every text gets the same embedding. Its code also needs a small shim for transformers 5. bf16 checkpoints are converted to fp32, because bf16 runs 9–25× slower on this CPU.

## 7. Reaching 95%: search the agent's own toolset, then Mercury Decide

Section 6 still left the right tool outside the top 100 for about a third of the steps. This section asks what it takes to keep it in at least 95% of the time, then runs Mercury Decide on top. `results/prefilter/deploy_*/report.md`, `results/deploy_mercury/report.md`.

**Why the pooled catalog cannot get there.** ToolRet pools the toolsets of 35 separate datasets into three catalogs: web has 37,292 tools from 19 datasets. With the whole catalog in play, 95% single-tool recall needs K ≈ 13,600 for BM25 + MiniLM on all of ToolRet. Even the best tool-trained ensemble needs K ≈ 1,700 on code + customized. No decision model reads that many candidates.

**What the literature points to.**
- **Hierarchical server-then-tool routing.** [MCP-Zero](https://arxiv.org/abs/2506.01056) first narrows the search to the relevant servers, then ranks tools within them. On APIBank, plain query retrieval scores 71.6%, while agent requests with this routing reach 95–97% (top-1, their Table 1).
- **The deployment is the catalog.** In ToolRet every step's gold tools come from its own dataset's toolset: we checked this for all 7,961 steps (`SUBSET_TOOLSET` in `data.py`). That toolset is what an agent deployed for the dataset is connected to, like its set of MCP servers.
- **Duplicate tools.** [RoTBench](https://arxiv.org/abs/2401.08326) adds noise to tool *names* at five levels and leaves the documentation unchanged. ToolRet keeps every copy, so 99% of rotbench steps have about 38 tools with the gold tool's exact description. Exact-id recall there is a guess among identical tools. We report exact-id recall, and duplicate-aware recall in which a same-description copy from the same toolset also counts.

**Three scopes for the prefilter** (`HybridPrefilter(scope=...)`):
- **`pooled`:** the whole catalog, as before.
- **`routed3`:** first the tools of the 3 toolsets whose best tool matches best (MCP-Zero-style routing, no labels).
- **`toolset`:** first the tools of the step's own dataset toolset, i.e. the agent's connected tools.

In all three, remaining slots are filled from the rest.

| catalogs (single-tool steps) | prefilter | scope | R@50 | R@100 | R@200 |
|---|---|---|---|---|---|
| code + customized (1,936) | BM25 + MiniLM (current) | pooled | 0.618 | 0.687 | 0.749 |
| | Tool-Embed + bge-ToolRet + BM25, expanded docs | pooled | 0.731 | 0.796 | 0.865 |
| | same | routed3 (finds the right toolset 94% of the time) | 0.732 | 0.801 | 0.849 |
| | BM25 + MiniLM, expanded docs | toolset | 0.830 | 0.895 | 0.944 |
| | **Tool-Embed + bge-ToolRet (dense RRF), expanded docs** | **toolset** | **0.899** | **0.947** | **0.972** |
| all of ToolRet (4,402) | BM25 + MiniLM, expanded docs | pooled | 0.672 | 0.721 | 0.766 |
| | same | toolset | 0.818 | 0.866 | 0.904 |
| | same, duplicate-aware | toolset | 0.838 | 0.905 | **0.945** |

- **Scoping to the agent's toolset is the step change:** +0.15 to +0.18 R@50 for every encoder. Zero-shot routing does not get there, because similar tools exist in other toolsets. On customized it finds the right toolset 65% of the time with one guess; with three, 93–94% on code + customized and 86% on all of ToolRet.
- **With the tool-trained ensemble, 95% single-tool recall needs K = 110 and 97% needs K = 190.** The remaining misses are concentrated in gorilla-huggingface (907 similar model APIs) and craft-vqa, both about 0.93 at K = 200. Every other toolset is at 0.95 or above by K = 150.
- **Across all of ToolRet with only MiniLM,** the toolset scope reaches 0.945 at K = 200 once RoTBench's copies count. The web catalog was not run with the tool-trained encoders: their training queries overlap 10% of web steps (section 6), and encoding its 37k tools takes about 3 h on this CPU.
- Multi-tool steps, where every gold tool must be found, stay much lower (0.50 at K = 200 on code + customized). Mercury's choice question picks one tool, so this section uses single-tool steps.

**End to end with Mercury Decide.** `scripts/run_deploy_mercury.py` takes the toolset-scoped ensemble over expanded docs and picks K = 200, the smallest K whose recall on the 1,886 other single-tool steps reaches 0.95 + 0.02 (0.971). It then sends a stratified sample of 50 steps to Mercury Decide, one choice question over the 200 candidates each. Conformal sets are leave-one-out: each step's threshold is calibrated on the other 49, and a prefilter miss counts as a failure.

| 50 single-tool steps, K = 200 | result |
|---|---|
| right tool among the 200 candidates (prefilter) | **0.98** (49 of 50; 0.971 on the 1,886 held-out steps) |
| Mercury top-1 / top-5 / top-10 / top-20 | 0.46 / 0.60 / 0.72 / 0.80 |
| conformal set at α = 0.05: coverage, avg size | **0.96**, 177 tools (median 200) |
| α = 0.10 | 0.90, 63 tools (median 28.5) |
| α = 0.20 | 0.80, 30 tools (median 6.5) |
| cost per call | about 16k input tokens, 1.8 s median, 0 refusals |

- **The prefilter target is met.** The right tool is among the candidates for 98% of the steps, and for 97% of the held-out steps.
- **The end-to-end guarantee holds, but the set is not yet useful.** At α = 0.05 the set contains the right tool 96% of the time (target 95%), but only by keeping about 177 of the 200 tools. Mercury's ranking is now the limit, not the prefilter.
- **Why Mercury ranks the labelled tool low here.** Several tools can do the job but ToolRet labels one of them:
  - **gorilla-huggingface (3 of 13 top-1):** Mercury picked `svalabs/gbert-large-zeroshot-nli` for a German zero-shot classifier; the label is `Sahajtomar/German_Zeroshot`.
  - **toolink:** Mercury picked `calculate_median`; the label is `find_median`.
  - **toolalpaca:** Mercury picked `checkReputationScore`; the label is `checkIPReputation`.
  - **toolbench-sam (0 of 5 top-1):** the label is the API that must be called first (`set_location`, `select_booking_type`), and Mercury picks another API the task also needs.
  - By contrast, craft-math-algebra is 7 of 7 top-1.
- **Mixing Mercury with the retriever's scores does not help.** The retriever's temperature was fitted on the 1,886 held-out steps. The blend's top-20 is 0.70 against 0.80 for Mercury alone, and its α = 0.1 set is 73 tools against 63.
- **This mix of catalogs is harder than section 4's.** In section 4 (mostly web steps, K = 50), Mercury's top-1 was 0.75 when the tool reached it; here it is 0.47, with 4× more candidates.
- **What would bring the set close to 100% at a useful size:**
  - judge whether the chosen tool is equivalent to the labelled one, instead of requiring the exact id (none of the misses above has an identical description);
  - fewer, sharper candidates per call through agent-written tool requests (MCP-Zero);
  - more calibration steps. α = 0.02 needs at least 49 calibration points per step, and a prefilter recall above 0.98.

## 8. Prefilter at K = 50: example requests per tool, then Mercury Decide

Section 7 needed K = 110 for 95% recall, and Mercury read 200 candidates (about 16k input tokens per call). This section asks what keeps the right tool in the top 50 for 95% of single-tool steps, then reruns Mercury at K = 50 on the same 50 steps. `scripts/run_prefilter_k50.py` measures recall within each step's own toolset, with no API calls. `results/prefilter_k50/report.md`, `results/deploy_mercury_k50/report.md`.

**Where section 7's prefilter misses at K = 50.** Ranked within the toolset, it misses 202 of 1,936 steps, and 167 of the misses are in three datasets:

| dataset | misses at K = 50 | what the request leaves open |
|---|---|---|
| gorilla-huggingface | 90 of 500 | which of 907 Hugging Face models: "identify cats, dogs or birds in images" fits many classifiers, and the label is one CLIP checkpoint |
| craft-vqa | 51 of 200 | which of 525 generated helper functions, e.g. `identify_attribute_type` vs `identify_attribute_of_associated_object` |
| craft-tabmwp | 26 of 174 | the table type (stem-and-leaf or frequency table), which is not in the request |

**What the literature suggests** (each number checked against the paper):

| approach | reported result | tested here |
|---|---|---|
| Represent tools by requests that used them, not only by their description: [Tool2Vec](https://arxiv.org/abs/2409.02141) averages the embeddings of a tool's past queries | AWSBank Recall@7: description-based retriever 49.13 → Tool2Vec 67.98 (their Table 2) | yes: per-tool example requests, and past requests of the same toolset |
| Synthetic example requests per tool: [Re-Invoke](https://arxiv.org/abs/2408.01875) generates queries from each tool's doc and indexes them | ToolE single-tool nDCG@5 0.652 → 0.781 with the query generator alone (their Table 3a) | yes, through Gorilla's training requests (below) |
| Generic cross-encoder rerankers ([ToolRet](https://arxiv.org/abs/2503.01763), reranking NV-Embed-v1) | average N@10 33.83 for the retriever; bge-reranker-v2-m3 29.24, mxbai-rerank-large-v1 24.84, jina-reranker-v2 33.60, bge-reranker-v2-gemma 35.51 (their Table 4) | yes: MiniLM-L6 and bge-reranker-base |
| Tool-trained rerankers: Tool-Rank-4B ([Tool-DE](https://arxiv.org/abs/2510.22670)); trained cross-encoder ([ToolRerank](https://arxiv.org/abs/2403.06551)) | N@10 52.23 → 56.44 over Tool-Embed-4B (Tool-DE Table 3); ToolBench Recall@5 75.2 → 80.3 over DPR (ToolRerank Table 3) | no: 4B parameters, or in-domain training, on 4 CPU cores |
| More tool-trained embedders: ToolRet-trained bge-large and e5-large | average N@10 32.95 and 30.18, against 23.03 and 20.78 untrained (ToolRet Table 14) | yes |
| Contrastive fine-tuning on tool-retrieval data | ToolRet-train (built from ToolACE, ToolBench and APIGen) lifts bge-base from 22.06 to 29.35 average N@10 (ToolRet Table 14); that model is already in the section 7 prefilter, and no cited paper measures further fine-tuning on the target catalog | in-domain, cross-fitted: started, not finished (one of five folds took over 55 minutes on this CPU) |
| Adaptive set size from calibrated thresholds on refined scores ([score refinement for conformal retrieval](https://arxiv.org/abs/2410.02914)) | FIQA, α = 0.05: fixed top-K needs 259 documents, refined-score thresholds 190.5, at the same 0.92 coverage; thresholds on raw scores need 846 (their Table 2) | yes, on raw scores: per toolset and per request |

**Signals tested.**
- **Gorilla's training requests.** [APIBench](https://arxiv.org/abs/2305.15334) writes 10 requests per API by self-instruct from the API's doc and holds out 10% for evaluation, so its training split has about 9 per Hugging Face model. ToolRet's gorilla-huggingface requests come from that eval split (452 of 500 match it exactly). None of the 8,131 training requests equals a ToolRet request, and the 281 within 0.95 cosine of one were dropped. Each tool is scored by the cosine to the mean embedding of its requests (Tool2Vec), then fused with RRF.
- **Past requests of the same toolset.** Query-to-query kNN over the other folds of a 5-fold split, with same-text requests left out. The gold tool had been used before for 39% of steps.
- **Fusion weights.** Every fusion weight and similarity floor is chosen on the other folds, never on the steps being scored. When the past-request gate is chosen for a fold, that fold is also left out of the other folds' request history.

| prefilter (toolset scope, 1,936 single-tool steps) | R@20 | R@50 | R@100 | R@200 | K for 95% | ΔR@50 [95% CI] |
|---|---|---|---|---|---|---|
| section 7 prefilter (Tool-Embed-0.6B + ToolRet-trained bge-base, dense RRF, expanded docs), ranked within the toolset | 0.794 | 0.896 | 0.948 | 0.973 | 110 | – |
| + ToolRet-trained bge-large | 0.796 | 0.901 | 0.956 | 0.977 | 89 | +0.006 [−0.002, +0.014] |
| + ToolRet-trained e5-large | 0.802 | 0.913 | 0.952 | 0.978 | 95 | +0.018 [+0.010, +0.026] |
| + past requests, plain RRF | 0.683 | 0.863 | 0.925 | 0.962 | 159 | −0.033 [−0.046, −0.021] |
| + past requests, weight and similarity floor from other folds | 0.815 | 0.909 | 0.949 | 0.973 | 106 | +0.013 [+0.007, +0.020] |
| + Gorilla training requests | 0.841 | 0.933 | 0.973 | 0.988 | 61 | +0.037 [+0.028, +0.046] |
| + e5-large + Gorilla training requests | 0.848 | 0.949 | 0.974 | 0.990 | 52 | +0.054 [+0.043, +0.065] |
| **+ e5-large + Gorilla training requests + past requests** | **0.864** | **0.951** | **0.974** | **0.990** | **50** | **+0.055 [+0.044, +0.067]** |

- **The target is met.** 95% single-tool recall now needs K = 50 instead of 110. With K fixed on the other folds, coverage is 0.951 at K = 49.6 on average. (Section 7's table gives 0.899 at K = 50 for the same prefilter; there, ranks run over the whole catalog with the toolset first.)
- **Where the gain comes from.** Recall@50 by dataset, section 7 prefilter → final:
  - gorilla-huggingface 0.820 → 0.972: the example requests (e5-large alone reaches 0.834);
  - toolbench-sam 0.901 → 0.995: e5-large to 0.963, then past requests;
  - craft-vqa 0.745 → 0.800 and metatool 0.972 → 0.989: e5-large;
  - craft-tabmwp 0.851 → 0.839: slightly worse.
- **Example requests beat descriptions.** On gorilla-huggingface, scoring tools by their example requests alone gives R@50 = 0.974, against 0.834 for the three description encoders (`Gorilla train requests only` in the report).
- **Past requests need a gate.** Added with plain RRF, they lower recall (R@20 0.794 → 0.683). With a similarity floor of 0.7–0.8 and a weight of 0.25–0.5, both chosen on the other folds, they help.

**What did not help.**
- **Small cross-encoder rerankers do not help here.** Reranking alone lowers recall, and RRF with the prefilter changes nothing significant. In ToolRet most generic rerankers also lowered N@10; bge-reranker-v2-gemma (2B parameters) raised it from 33.83 to 35.51. This is on a stratified sample of 150 steps, reordering the top 100 of the best prefilter (R@50 0.947 on this sample):

  | reranker | rerank alone: R@50, Δ [95% CI] | RRF with the prefilter: R@50, Δ [95% CI] |
  |---|---|---|
  | ms-marco-MiniLM-L6-v2 | 0.867, −0.080 [−0.133, −0.027] | 0.940, −0.007 [−0.033, +0.013] |
  | bge-reranker-base | 0.880, −0.067 [−0.113, −0.020] | 0.953, +0.007 [−0.013, +0.033] |

  On this CPU the base-size reranker scores about 7 request–tool pairs per second.
- **Adaptive K saves little at the same 95%.** A separate K per toolset gives a median of 38 tools, but the mean is 60, because craft-tabmwp needs about K = 110 and craft-vqa more than 200 (gorilla-huggingface now needs about 38). A per-request score margin calibrated per toolset behaves the same way (median 36, mean 62). A fixed K = 50 is simpler and as small on average.
- **CRAFT cannot use its own data.** All 174 ToolRet craft-tabmwp requests are exactly the training problems CRAFT abstracted its tools from, so using that data would leak the label. craft-vqa (0.800) and craft-tabmwp (0.839) stay the weakest datasets.

**Caveats.**
- **Most of the gain needs example requests per tool:** logged requests, or synthetic ones as in Re-Invoke. Without them, the best is the e5-large ensemble, at 0.913.
- **Methods were chosen on the same benchmark.** Weights were picked on other folds, so each step's estimate is held out, but which signals to combine was decided by looking at this benchmark.
- **Code + customized only.** The web catalog was not run.

**Mercury Decide at K = 50.** `scripts/run_deploy_mercury.py --candidates results/prefilter_k50/candidates.npz --k 50` sends the same 50 steps as section 7, now with the top 50 of the new prefilter. Conformal sets are leave-one-out, as before. This run used the ranking before the gate fix above, in which the held-out fold's requests could still influence its past-request gate. In the final ranking 13 of these 50 steps have a different candidate order and 3 a different candidate set; recall on the sample is 0.94 in both.

| same 50 single-tool steps | K = 200 (section 7) | K = 50 (this section) |
|---|---|---|
| right tool among the candidates | 0.98 (0.971 held out) | 0.94 (0.952 held out) |
| Mercury top-1 / top-5 / top-10 / top-20 | 0.46 / 0.60 / 0.72 / 0.80 | 0.44 / 0.66 / 0.78 / 0.86 |
| conformal set at α = 0.05: coverage, avg size | 0.96, 177 tools | 0.94, 50 tools (all candidates) |
| α = 0.10 | 0.90, 63 tools (median 28.5) | **0.90, 35 tools** (median 39.5) |
| α = 0.20 | 0.80, 30 tools (median 6.5) | **0.80, 9.3 tools** (median 3) |
| input tokens per call, median latency | about 16k, 1.8 s | **about 4.1k, 1.0 s** |
| prefilter scores alone (softmax, no Mercury): top-1, α = 0.10 set | 0.38, 85 tools | 0.44, 39 tools |

- **Same coverage, smaller sets on average, a quarter of the cost.** At α = 0.10 and 0.20 the K = 50 sets reach the same coverage with about half and a third of the tools on average. At α = 0.10 the median set grows, though, from 28.5 to 39.5 tools: the mean falls because no set can exceed 50. Each call reads 74% fewer tokens.
- **α = 0.05 is out of reach on this sample.** The prefilter misses 3 of the 50 steps, so coverage stops at 0.94; on the held-out steps it is 0.952.
- **Mercury's accuracy on the steps it sees is unchanged.** Top-1 is 0.47 in both runs when the right tool is among the candidates. Top-10 and top-20 rise by 3 steps each with fewer distractors.
- **The new prefilter alone now ranks as well as Mercury at top-1** (0.44 for both on these 50 steps), and its α = 0.10 set is 39 tools.
- **Confidence is low.** At n = 50 a difference of 0.02 is one step. None of the top-k or coverage differences (3 steps or fewer) is significant; the set-size differences were not tested.

## 9. Stress tests (RQ3/RQ4), hybrid score source, ~1,080 single-tool test steps

`results/minilm/stress/report_hybrid.md`

| test | static split CP | online ACI |
|---|---|---|
| clean | 0.886 | – |
| catalog drift (−20% tools, +20% foreign tools, 20% renamed, lookalikes for half the gold tools) | 0.910 | 0.904 |
| shift: calibrate on web, test on code + customized | **0.826** | **0.900** |
| shift: calibrate on code, test on web + customized | 0.950 (over-covers) | 0.903 |

- **H3:** under distribution shift, split CP misses its target, and ACI returns to 0.90 in both directions.
- **Drift:** it does not hurt this scorer. Its sets are large (about 20 tools), so lookalikes ride along (they appear in 54% of sets) without displacing the gold tool.
- **H4 (poisoned descriptions):**
  - At 1% poisoned, the poisoned tools' inclusion rate rises from 0.145 to 0.177, and the sanity check restores it to 0.148.
  - At 5%, there is no measurable effect.
  - Detector: recall 1.00, precision 0.84–0.97, false positives on 0.19% of clean tools.

  Embedding/RRF scores barely react to injected text. The case that matters is the decision model, which reads descriptions as instructions; that run is still to do.

## 10. Not yet run

| run | needs |
|---|---|
| Phase 1 at scale with Mercury (Mode A + B, ~1,000 steps) | Mercury Decide, one call per step per mode |
| Stress tests with Mercury (lookalikes and poisoning against a sharp scorer) | about 1,200 Mercury Decide calls |
| Phase 2 agent-in-the-loop proxy (RQ2) | an agent chat model (OpenAI key or an OpenRouter chat model) |
| Phase 4 governance latency (RQ5) | 2 Mercury Decide calls per step |
