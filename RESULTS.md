# Results so far (10 Oct 2026)

All runs use ToolRet (7,961 steps, 44,453 tools). The Stage-1 prefilter is BM25 + `multi-qa-MiniLM-L6-cos-v1` fused with RRF, K = 50. The decision model is Inception Mercury Decide on OpenRouter (free tier). Detailed tables and figures are in each `results/**/report.md`.

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

End to end, q̂ = ∞ for every α ≤ 0.2 and every score source, so the set is all 50 tools. Coverage is then capped at recall: **0.62** for single-tool steps and 0.28 for multi-tool steps. No Stage-2 scorer can fix this, so a stronger prefilter is the main lever. Options: Qwen3-Embedding-0.6B (encoding paused at 8k/44k tools, resumable with `scripts/warm_embeddings.py`) or toolrank.

## 4. Decision model smoke test (H1), 47 Mercury Decide calls

59 single-tool steps (12 dev + 47 eval); Stage-2 view n = 28. `results/mercury_smoke/phase1/report.md`

| score source | top-1 acc. | coverage at α = 0.1 | avg tools shown (of 50) | coverage if capped at 5 |
|---|---|---|---|---|
| cross-encoder | 0.28 | 0.94 | 28.2 | 0.64 |
| **Mercury Decide** | **0.45** | **0.94** | **2.3** | **0.94** |

- **At matched coverage, Mercury's sets are about 12× smaller than the cross-encoder's.** Mercury against dense: −40.8 tools, 95% CI [−42.3, −39.3]. This is the direction H1 predicts. n = 28, so treat it as a strong signal, not a result.
- **Mercury is overconfident:** ECE 0.26, with confidence near 1.0 but accuracy near 0.7. Conformal calibration absorbs this; a raw confidence threshold would not.
- **Usage:** median API latency 1.1 s, about 2.7k input tokens per call, $0.

## 5. Stress tests (RQ3/RQ4), hybrid score source, ~1,080 single-tool test steps

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

  Embedding/RRF scores barely react to injected text. The case that matters is the decision model, which reads descriptions as instructions; that run needs API quota.

## 6. Not yet run (needs quota or keys)

| run | needs |
|---|---|
| Phase 1 at scale with Mercury (Mode A + B, ~1,000 steps) | OpenRouter 1,000/day tier ($10 credit) or about 20 days on the free 50/day |
| Stress tests with Mercury (lookalikes and poisoning against a sharp scorer) | about 1,200 calls |
| Phase 2 agent-in-the-loop proxy (RQ2) | an agent chat model (OpenAI key or an OpenRouter chat model) |
| Phase 4 governance latency (RQ5) | 2 calls per step; 20 steps fits the free tier |
