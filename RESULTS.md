# Results so far (10 Oct 2026)

All runs use ToolRet (7,961 steps, 44,453 tools). The Stage-1 prefilter is BM25 + `multi-qa-MiniLM-L6-cos-v1` fused with RRF, K = 50 (section 5 varies K; section 6 tests stronger prefilters). The decision model is Inception Mercury Decide on OpenRouter (free tier). Detailed tables and figures are in each `results/**/report.md`.

## 1. Tests

`pytest`: 31 tests pass. They check that:

- **Coverage guarantee (synthetic data):** split conformal hits 1 − α for LAC and APS.
- **Risk control:** conformal risk control keeps the false-negative rate ≤ α, and its exact threshold matches brute force.
- **Prefilter misses:** they push q̂ to ∞, and coverage is then capped at recall@K.
- **Online ACI:** it restores coverage after a score shift.
- **API clients:** OpenRouter, OpenAI Decisions and Chat Completions work against mocked HTTP endpoints, including retry on 429 and caching.
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

## 7. Stress tests (RQ3/RQ4), hybrid score source, ~1,080 single-tool test steps

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

## 8. Not yet run

| run | needs |
|---|---|
| Phase 1 at scale with Mercury (Mode A + B, ~1,000 steps) | Mercury Decide, one call per step per mode |
| Stress tests with Mercury (lookalikes and poisoning against a sharp scorer) | about 1,200 Mercury Decide calls |
| Phase 2 agent-in-the-loop proxy (RQ2) | an agent chat model (OpenAI key or an OpenRouter chat model) |
| Phase 4 governance latency (RQ5) | 2 Mercury Decide calls per step |
