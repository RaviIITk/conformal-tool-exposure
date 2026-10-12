# Conformal Tool Exposure

Coverage-guaranteed tool selection for LLM agents. Agents connected to many MCP servers cannot fit every tool description into context, and showing a fixed top-k gives no guarantee that the right tool is among them. Given what an agent needs to do next, this project exposes the smallest set of tools that contains the right one with probability ≥ 1 − α.

The study design is in [`research_design_conformal_tool_exposure.md`](research_design_conformal_tool_exposure.md), and the tooling survey in [`requirements.md`](requirements.md). The pipeline has three steps:

1. **Stage 1 prefilter:** BM25 + dense embeddings, fused with Reciprocal Rank Fusion, keep the top K candidates.
2. **Stage 2 score source:** gives each candidate a probability. The options are:
   - local sources: BM25, dense embeddings, a cross-encoder;
   - **the decision model:** Inception Mercury Decide, through OpenRouter's Jev-compatible Decisions API (free tier). The same client also serves TypeSafe Jev and Cloudflare Clef;
   - optionally, the OpenAI Decisions API and an LLM log-probability baseline.
3. **Conformal set:** split conformal prediction (LAC, APS), conformal risk control for multi-tool steps, and online adaptive conformal (ACI) for drift.

The whole pipeline is wired into a LangChain agent as middleware: the agent sees only `find_tool`, then only the conformal set.

## Results

On the [ToolRet](https://arxiv.org/abs/2503.01763) benchmark (7,961 requests, 44,453 tools), with Mercury Decide as the Stage-2 scorer: when the agent is shown k tools, the right tool is among them this often (requests where the right tool reached Stage 2):

```
tools shown to agent     1     2     3     5
right tool included     75%   86%   93%   96%
```

With adaptive set sizes (target 90%), the agent sees 2.3 tools on average and the right tool is included 94% of the time. The best local scorer, a cross-encoder, needs about 28 tools for the same coverage. The coverage guarantee itself was validated on the full benchmark with local scorers. The main open problem is Stage-1 recall: the prefilter keeps the right tool among 50 candidates for 62% of requests.

**Prefilter K sweep** (same 47 requests, prefilter K varied, Mercury Decide and dense scorers; set sizes are for requests where the right tool reached Stage 2):

```
prefilter K                              10     20     30     50    100
right tool kept by prefilter            43%    49%    55%    60%    68%
Mercury: right tool ranked 1st (e2e)      –      –      –    45%    47%
Mercury: tools shown at 90% target        –      –      –    2.3    5.0
dense:   tools shown at 90% target      8.7   16.2   26.5   43.1   83.7
```

Raising K from 50 to 100 adds 8 points of prefilter recall but only one more correct top-1, and doubles Mercury's set size. Mercury's accuracy on the steps it already saw is unchanged; the added steps are harder. A larger K alone does not fix the prefilter.

**Better prefilters** (literature review and tests, RESULTS.md section 6). Recall@50 on ToolRet's code + customized catalogs (2,731 requests), which the tool-trained models were not trained on:

```
prefilter                                    recall@50   vs current        CPU ms/query
BM25 + MiniLM (current)                        0.504         –                   9
  + expanded tool docs                         0.525     +0.021                  9
BM25 + LFM2.5-Embedding-350M + expanded        0.530     +0.026                158
ToolRet-trained bge-base + expanded            0.560     +0.056                 57
Tool-Embed-0.6B + expanded                     0.593     +0.089                355
```

Tool-trained embedders ([ToolRet](https://arxiv.org/abs/2503.01763), [Tool-DE](https://arxiv.org/abs/2510.22670)) help most. Document expansion (LLM-written tool profiles, built offline) helps every prefilter at no query-time cost. General embedders help little on their own: LFM2.5-350M adds +0.010 (not significant), and Nemotron-3-Embed-1B adds +0.031 on the customized catalog at 488 ms per query. Agent-written tool requests are the largest remaining lever: ToolRet's instruction adds +0.107 on all of ToolRet, though that is an upper bound.

**Reaching 95%** (RESULTS.md section 7). ToolRet pools 35 datasets' toolsets into one catalog. Searching only the toolset the agent is actually connected to (its own dataset's tools, as in MCP-Zero's server-then-tool routing) is the step change. With a Tool-Embed + bge-ToolRet ensemble over expanded docs, the right tool is among K = 200 candidates for 97% of single-tool requests on code + customized (K = 110 for 95%; the whole catalog would need K ≈ 1,700). End to end with Mercury Decide on 50 of these requests:

```
right tool among the 200 candidates          98%
Mercury top-1 / top-5 / top-20                46% / 60% / 80%
conformal set, alpha = 0.05                   96% coverage, 177 tools on average
conformal set, alpha = 0.10                   90% coverage, 63 tools (median 28)
```

The prefilter is no longer the limit; Mercury's ranking is. On these catalogs several tools can do the job and ToolRet labels only one (e.g. two German zero-shot classifiers, `find_median` vs `calculate_median`), so a 95% set still has to keep most candidates.

**Prefilter at K = 50** (RESULTS.md section 8). Ranking each tool also by example requests that used it (as in [Tool2Vec](https://arxiv.org/abs/2409.02141) and [Re-Invoke](https://arxiv.org/abs/2408.01875)) keeps the right tool in the top 50 for 95% of single-tool requests on code + customized:

```
prefilter (agent's own toolset)                            recall@50   K for 95%
Tool-Embed + bge-ToolRet, expanded docs (section 7)*         0.896        110
  + ToolRet-trained e5-large                                 0.913         95
  + Gorilla's training requests per Hugging Face model       0.949         52
  + similar past requests of the same toolset                0.951         50
generic cross-encoder rerankers on top (150-step sample)     no gain, or worse
```

\* ranked within the toolset; section 7's own table gives 0.899.

The example requests are Gorilla's training split ([APIBench](https://arxiv.org/abs/2305.15334)), which shares no exact request with ToolRet (281 near-duplicates, cosine ≥ 0.95, were dropped); past requests come from other cross-fitting folds, and every fusion weight is chosen on other folds. craft-vqa (0.80) and craft-tabmwp (0.84) remain the weakest datasets. Mercury Decide on the same 50 requests, K = 200 vs K = 50 (the K = 50 run used the ranking before a last fix to the past-request gate, which changed 3 of these 50 candidate sets):

```
                                          K = 200        K = 50
right tool among the candidates             98%            94%
Mercury top-1 / top-10                    46% / 72%      44% / 78%
conformal set, alpha = 0.10               90%, 63 tools  90%, 35 tools
conformal set, alpha = 0.20               80%, 30 tools  80%, 9 tools
input tokens per call, median latency     16k, 1.8 s     4.1k, 1.0 s
```

Same coverage with about half the tools on average at 90% (the median rises from 28.5 to 39.5) and a third at 80%, at a quarter of the input tokens; 95% is out of reach on this sample because the prefilter misses 3 of the 50 requests. Full numbers are in [`RESULTS.md`](RESULTS.md) and `results/`.

## Setup

```bash
uv venv .venv --python 3.12
```

```bash
uv pip install --python .venv/bin/python -e ".[dev]"
```

ToolRet (about 16 MB) downloads into `data/toolret/` on first run. Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`, which the decision model needs. `OPENAI_API_KEY` is optional.

One decision-model request scores one agent step in either mode. Responses are cached, so a run can be resumed. Set `CTE_MAX_API_CALLS=N` to make a run fail rather than exceed N new calls.

## Run

| What | Command | Needs key |
|---|---|---|
| Tests (synthetic coverage guarantee, mocked API endpoints, middleware flow) | `.venv/bin/python -m pytest` | no |
| Phase 1, RQ1/H1: coverage and set size per score source | `.venv/bin/python scripts/run_phase1.py` | no |
| Phase 1 with the decision model (Mercury Decide; 47 calls with these flags) | `.venv/bin/python scripts/run_phase1.py --max-steps 59 --single-tool-only --sources bm25,dense,hybrid,ce,mercury-choice` | OpenRouter |
| Phase 1 at scale | `.venv/bin/python scripts/run_phase1.py --max-steps 1200 --sources bm25,dense,hybrid,ce,ce-pred,mercury-choice,mercury-predicate` | OpenRouter |
| Prefilter K sweep: prefilter recall and final results (dense + Mercury Decide) at each K; `--api-k-values` picks the K values Mercury runs at | `.venv/bin/python scripts/run_k_sweep.py --max-steps 59 --single-tool-only --encoder sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --k-values 10,20,30,50,100 --api-k-values 100` | OpenRouter |
| Prefilter benchmark: recall@K of BM25, embedders and ensembles (`A+B`), expanded tool docs, scopes (`--scopes pooled,routed3,toolset`) (no API calls) | `.venv/bin/python scripts/run_prefilter_bench.py --tag code_customized --categories code,customized --queries query --encoders sentence-transformers/multi-qa-MiniLM-L6-cos-v1,Lux1997/Tool-Embed-0.6B` | no |
| End to end with the agent's own toolset as the prefilter, then Mercury Decide (one call per step) | `.venv/bin/python scripts/run_deploy_mercury.py --n 50` | OpenRouter |
| Prefilter at small K: more tool-trained encoders, example requests per tool (Gorilla's training split), past requests, rerankers (`--rerankers`), adaptive K; saves the best ranking to `candidates.npz` (no API calls) | `.venv/bin/python scripts/run_prefilter_k50.py --rerankers cross-encoder/ms-marco-MiniLM-L6-v2,BAAI/bge-reranker-base` | no |
| Mercury Decide on the small-K prefilter's top 50 | `.venv/bin/python scripts/run_deploy_mercury.py --candidates results/prefilter_k50/candidates.npz --k 50 --k-grid 50,100,150,200 --out results/deploy_mercury_k50` | OpenRouter |
| Phase 3, RQ3/RQ4: drift, shift, poisoned descriptions | `.venv/bin/python scripts/run_stress.py --source ce` | no (OpenRouter with `--source mercury-choice`) |
| Phase 2 proxy, RQ2: agent success per token by exposure condition | `.venv/bin/python scripts/run_phase2_proxy.py --max-steps 1500 --n-steps 300` | OpenRouter + agent chat model |
| Phase 4, RQ5: latency of adding risk questions to the same call | `.venv/bin/python scripts/run_governance.py --max-steps 1500 --n-steps 20` | OpenRouter |
| LangChain agent demo | `.venv/bin/python scripts/demo_agent.py` | no (scripted agent); yes for a real agent |

To run any phase with the stronger prefilter, add `--encoder Lux1997/Tool-Embed-0.6B+mangopy/ToolRet-trained-bge-base-en-v1.5 --no-bm25 --tool-docs expanded --prefilter-scope toolset`. Each script writes `results/<phase>/report.md`, JSON and figures. Every API response is cached in `cache/*.jsonl`, so reruns cost nothing. On a slow network, `HF_HUB_DISABLE_XET=1` makes Hugging Face downloads use plain HTTPS.

Cost: Mercury Decide's `:free` variant costs $0; Phase 1 sends about 2.7k input tokens per call at K = 50. If you use the OpenAI Decisions API instead, 1,500 steps cost about $0.45 per mode at $0.10/M input tokens.

## Layout

| File | Design section |
|---|---|
| `src/cte/data.py` | ToolRet catalogs and steps (§9) |
| `src/cte/retrieval.py` | Stage 1 hybrid prefilter (BM25 + encoder ensembles, RRF; pooled, routed or own-toolset scope), recall@K (§7.4) |
| `src/cte/scorers.py` | local score sources with temperature/Platt calibration (§10.2) |
| `src/cte/openai_scorers.py` | Jev-compatible decision-model client (OpenRouter Mercury Decide; Mode A `choice`, Mode B `noul`), OpenAI Decisions API, log-prob baseline (§7.5) |
| `src/cte/conformal.py` | LAC, APS, conformal risk control, end-to-end +∞ scores, k_max cap, fallback, ACI (§8) |
| `src/cte/metrics.py` | coverage, set size, ECE, Beta coverage theory, paired bootstrap, Holm (§11–12) |
| `src/cte/stress.py` | drift, poisoning and description-sanity defence (§10.4) |
| `src/cte/middleware.py` | `find_tool` + `wrap_model_call` + `wrap_tool_call` governance (§7.1, 7.3, 7.6) |
| `src/cte/agent_eval.py` | Phase 2 single-step agent proxy (§10.3) |

## Deviations from the design

- **Decision model.** It is Inception Mercury Decide on OpenRouter, a hosted Jev-compatible model, not self-hosted Clef-flash, which does not fit on the 16 GB laptop (Apple M1) used here. Jev and Clef plug into the same `SystemOneSource` by changing `CTE_SYSTEMONE_URL` and `CTE_SYSTEMONE_MODEL`.
- **Stage 1 uses BM25 + a small open embedder** instead of toolrank (8B), for the same hardware reason. This lowers prefilter recall, which is why every table shows both end-to-end and Stage-2-only results.
- **`find_tool` is a real tool that updates agent state**, not an `after_model` jump. This avoids the unverified jump behaviour.
- **Phase 2 is a single-step proxy on ToolRet**, not live MCP servers. Condition A (all tools) is infeasible at these catalog sizes; condition C (agent-driven discovery) needs the live multi-step loop.
- **RQ5 measures only the latency overhead.** Attack success and false-block rates need AgentDojo tasks.
