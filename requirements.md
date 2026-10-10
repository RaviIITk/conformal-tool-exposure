# Requirements and Open-Source Availability

**For:** `research_design_conformal_tool_exposure.md` (v0.1) · **Checked:** 9 October 2026 (web search) · **Goal:** run the study at minimum cost

**Status key**

- **Open**: open-source code or open weights, free to use (you pay only for compute)
- **Free tier**: hosted service whose free quota covers part of the work
- **Build**: no public implementation found, so you write it (every item here is small)
- **Paid**: closed or paid service
- **Unverified**: could not be confirmed

Vendor figures (latency, prices, limits) are self-reported or come from third-party guides. Re-check them before relying on them.

---

## 1. Summary

| Status | Count |
|---|---|
| Open | 27 |
| Free tier | 1 |
| Build (small) | 5 |
| Paid | 5 |
| Unverified | 1 |
| **Total** | **39** |

**27 of 39 requirements (69%) are available as open source, and 28 (72%) can be used for free.** Everything the core method needs (decision model, retrievers, harness, conformal prediction and datasets) is open. The only spend you cannot avoid is the closed frontier agent LLM and the benchmark judge in Phase 2. Jev and the OpenAI Decisions API are optional comparisons. The 80 GB GPU can be skipped by using hosted Clef.

---

## 2. Requirements by area

### 2.1 Decision models (Stage-2 score sources)

| ID | Requirement | What is available | Licence / access | Cost | Status |
|---|---|---|---|---|---|
| R1 | Primary open decision model | Cloudflare **Clef-flash** 9B (`Cloudflare/clef-flash`), built on Qwen3.5-9B | Apache-2.0 weights; also on Workers AI | Self-hosted: $0 on a free GPU. Hosted: $0.09/M input tokens; free 10k Neurons/day ≈ 1.2M tokens | Open |
| R2 | Accuracy-oriented decision model | Cloudflare **Clef** 27B (`Cloudflare/clef`), built on Qwen3.8-27B | Apache-2.0; Workers AI | Hosted: $0.24/M. Self-hosting needs ~55 GB VRAM | Open |
| R3 | Extra open baseline | autotrust/JEV-9B and JEV-27B (unofficial students distilled from Jev) | Open weights on HF; community model | $0 (compute only) | Open |
| R4 | Closed reference model | TypeSafe **Jev** (`jev-1.13.0`) | Hosted only. Console signups paused 22 Sep 2026; still served by Vercel AI Gateway, OpenRouter and Cloudflare | $0.042/M input | Paid (optional) |
| R5 | Optional closed baseline | OpenAI Decisions API (GPT-6 Luna), `/v1/decisions` | Public beta since 6 Oct 2026 | $0.10/M input | Paid (optional) |

### 2.2 Stage-1 prefilter and comparison scorers

| ID | Requirement | What is available | Licence | Status |
|---|---|---|---|---|
| R6 | BM25 sparse retrieval | `bm25s`, `rank_bm25` or Pyserini (CPU only) | MIT / Apache-2.0 | Open |
| R7 | Reciprocal Rank Fusion | About 10 lines of code | — | Build |
| R8 | Tool-tuned dense retriever | **toolrank** 0.2.0 on PyPI; weights `yasinyaman/toolrank-emb-8b` (Qwen3-Embedding-8B + LoRA, 16.4 GB BF16, FP8 option). Its training data states no licence | Apache-2.0 | Open |
| R9 | General embedding model (ablation) | Qwen3-Embedding 0.6B/4B/8B or BGE-M3 | Apache-2.0 / MIT | Open |
| R10 | Cross-encoder reranker | bge-reranker-v2-m3 or Qwen3-Reranker | Apache-2.0 | Open |
| R11 | LLM log-probability scores | Any open LLM served with logprobs in vLLM. No public CP-OPT code found, so reimplement the learned-score step if needed | Open models | Open |

### 2.3 Agent harness, MCP and serving

| ID | Requirement | What is available | Licence | Status |
|---|---|---|---|---|
| R12 | Intercept `find_tool` and restrict tools per model call | LangChain v1 `AgentMiddleware`: `before_model` / `after_model`, `wrap_model_call` (`request.override(tools=...)`), `wrap_tool_call` | MIT | Open |
| R13 | Prebuilt tool-selection baseline | LangChain `LLMToolSelectorMiddleware` (`max_tools`, `always_include`) | MIT | Open |
| R14 | Alternative harness | Strands Agents SDK hooks (`BeforeModelCallEvent`, `AfterModelCallEvent`, `BeforeToolCallEvent`) and `ToolRegistry.register_dynamic_tool` / `replace`. Changes take effect on the next invocation; per-call tool filtering is not documented | Apache-2.0 | Open |
| R15 | MCP client and server plumbing | MCP Python SDK, `langchain-mcp-adapters` | MIT | Open |
| R16 | Model serving | vLLM, SGLang, HF Transformers (the Clef model cards give commands for each) | Apache-2.0 | Open |

### 2.4 Conformal prediction and statistics

| ID | Requirement | What is available | Status |
|---|---|---|---|
| R17 | Split CP with LAC and APS scores | MAPIE (BSD-3), TorchCP (LGPL-3.0), or about 30 lines of NumPy | Open |
| R18 | Conformal risk control (false-negative rate for multi-tool steps) | Reference code from Angelopoulos et al.; about 20 lines | Open |
| R19 | Online / adaptive CP for drift | TorchCP online module; the ACI update is a few lines | Open |
| R20 | End-to-end two-stage calibration (s = +∞ on a prefilter miss), k_max cap, empty-set fallback | No library; specific to this design | Build |
| R21 | Statistics and calibration metrics: bootstrap CIs, Holm correction, Beta coverage check, ECE, reliability diagrams | SciPy, statsmodels, NumPy, torchmetrics `CalibrationError` | Open |

### 2.5 Datasets and benchmarks

| ID | Requirement | What is available | Licence | Status |
|---|---|---|---|---|
| R22 | Large calibration and test pool | **ToolRet**: 7,961 queries, 44,453 tools (`mangopy/ToolRet-Queries`, `ToolRet-Tools`); training set also released. arXiv 2503.01763, ACL Findings 2025 | Code Apache-2.0; data licence not stated | Open |
| R23 | Agent-in-the-loop MCP benchmark | **LiveMCPBench**: 95 tasks, 70 servers, 527 tools, Docker image `hysdhlx/livemcpbench`. Annotations list the required tools and key points per task (task level, not per step) | Apache-2.0 | Open |
| R24 | Multi-step MCP benchmark | **MCP-Bench** (Accenture): 28 servers, 250 tools. Server API keys are free (NASA, NPS, Google Maps, HF, NCI). No per-step gold labels | Apache-2.0 | Open |
| R25 | Large catalog for scale and drift tests | **MCP-Zero** tools: 308 servers, 2,797 tools as JSON with precomputed embeddings. No queries are included | MIT | Open |
| R26 | Adversarial and prompt-injection tests | **AgentDojo** (`pip install agentdojo`) | MIT | Open |
| R27 | Decision-model sanity check | **Jev Decision Index** on HF: 132,422 requests across 37 benchmarks, 30+ open decision models | Public | Open |
| R28 | Calibration sanity dataset | TypeSafeAI/Open-Jev on HF (page returned 401 without login) | ? | Unverified |
| R29 | Silver labels, synthetic queries, tool equivalence classes | Generate with an open LLM (toolrank wrote MCP-Zero queries with Qwen3-8B, one per tool) | — | Build |

### 2.6 Agent LLMs and judges

| ID | Requirement | What is available | Status |
|---|---|---|---|
| R30 | Open-weight agent LLM | Qwen3-family or similar, served by vLLM | Open |
| R31 | Closed frontier agent LLM | Any frontier API | Paid |
| R32 | LLM judge for task success | MCP-Bench fixes o4-mini so results are reproducible. LiveMCPBench lets you configure the judge model, but its default setup needs an OpenAI key. An open judge works but makes results harder to compare with published numbers | Paid |

### 2.7 Baseline systems

| ID | Requirement | What is available | Status |
|---|---|---|---|
| R33 | Agent-driven discovery (Condition C) | MCP-Zero code, `xfey/MCP-Zero` (MIT) | Open |
| R34 | SCOUT-style gateway baseline | Paper only (arXiv 2608.23992); no code found. Rebuild from R6 + R7 + R8 | Build |
| R35 | RAG-MCP baseline | Only a community repo found (`memoverflow/rag-mcp`). Condition B (fixed top-k) already covers it | Build |

### 2.8 Compute and operations

| ID | Requirement | What is available | Status |
|---|---|---|---|
| R36 | GPU for Clef-flash (18.8 GB BF16) and toolrank-8B (16.4 GB BF16, about 8 GB FP8) | Kaggle gives about 30 GPU-hours/week free (2×T4 = 32 GB, or P100 16 GB); Colab also has a free tier. T4 and P100 lack BF16, so run in FP16 or quantized | Free tier |
| R37 | GPU for Clef 27B (54.7 GB BF16) | Rent an 80 GB GPU, or use hosted Clef on Workers AI instead | Paid (avoidable) |
| R38 | Logs, traces and metrics | JSONL files plus MLflow (Apache-2.0) or self-hosted Langfuse (MIT) | Open |
| R39 | Human audit of a 10% sample of silver labels | Label Studio Community (Apache-2.0) or a spreadsheet | Open (costs human time) |

---

## 3. Minimum-cost plan

| Phase | Cheapest setup | Estimated cost |
|---|---|---|
| Setup | Download ToolRet, MCP-Zero, LiveMCPBench, MCP-Bench and AgentDojo. Embed each catalog **once** with toolrank FP8 on Kaggle; store vectors in NumPy or FAISS | $0 |
| Phase 1 (RQ1) | Clef-flash self-hosted on Kaggle, or on Workers AI. Clef on Workers AI. Jev through OpenRouter or Vercel only if you have a key | For 10,000 calls × 8k tokens = 80M tokens: Clef-flash ≈ $7 (or $0 self-hosted), Clef ≈ $19, Jev ≈ $3.4, OpenAI Decisions ≈ $8 |
| Phase 2 (RQ2) | Open-weight agent on a free GPU for every condition. Run the closed frontier agent only on LiveMCPBench (95 tasks) for the key conditions B, D and E | Largest cost; size it after Phase 1 |
| Phase 3 (RQ3, RQ4) | Replay cached Phase 1 scores against perturbed catalogs offline. New agent runs are needed only for the multi-turn test | Mostly $0 |
| Phase 4 (RQ5) | Add risk questions to the same Clef-flash call; run AgentDojo with the open agent | Small |

**Without the frontier agent and judge, the total is roughly $0–40.**

Ways to keep cost down:

1. **Cache every decision-model score** per (situation, candidate list). All CP runs (α grid, 100+ random splits, drift replays) then reuse the cache at no extra cost.
2. Use the Workers AI free tier (about 1.2M Clef-flash tokens/day, roughly 150 calls at 8k tokens) for development and debugging only.
3. Token cost grows linearly with K. Use K = 20 or 50 unless the dev split shows that recall@K needs 100.
4. If you quantize Clef-flash to fit a free 16 GB GPU, calibrate and test with the **same** quantized model, and report it as quantized.

---

## 4. Gaps and updates to the design document

These answer the open items in Section 17 of the design document where the search found an answer.

1. **Jev limits (item 1).** A third-party guide lists up to 255 Choice options, about 32k tokens for state plus the longest question (about 64k shared across questions), 80 req/s and $0.042/M. The official docs page does not state these limits, so still verify them. Access risk is lower than the design implies: the console paused signups, but Jev is still reachable through Vercel, OpenRouter and Cloudflare.
2. **Clef limits (item 2).** Choice takes 2–255 options, a request can hold up to 64 questions, and the context is 65,536 tokens (third-party guide). The model cards set a default `max_length` of 16,384, which can be raised. So the answer is 64k, not 32k, and K = 100 fits.
3. **Per-step labels (item 3).** LiveMCPBench lists required tools per task, not per step. MCP-Bench has no per-step labels. Step-level calibration on either benchmark needs silver labels (R29).
4. **ToolRet (item 4).** The paper is arXiv 2503.01763 (ACL Findings 2025). The code is Apache-2.0; the data licence is not stated.
5. **LangChain tool selector (item 6).** It exists: `LLMToolSelectorMiddleware`. Add it as a Phase 2 baseline (an LLM picks a fixed top-k).
6. **Strands (item 7).** Tools can be added or replaced at runtime through `ToolRegistry`, effective on the next invocation. No documented way to filter tools per model call was found. Keep Strands for replication only.
7. **OpenAI Decisions API (item 8).** No longer a limited preview: public beta since 6 Oct 2026, $0.10/M input tokens.
8. **Not verified: `after_model` → `jump_to: "model"`.** The LangChain docs list `model` as a jump target, but the `after_model` example only shows `end`. Check this in code. Fallback: make `find_tool` a normal tool that writes the conformal set into agent state, and have `wrap_model_call` expose that set on the next turn. This needs no jump.
9. **Not verified: quantized Clef-flash.** Hugging Face lists community quantizations; it is not confirmed that they keep the decision head.
10. **No public code found** for SCOUT or CP-OPT (Prune 'n Predict). Both are small to reimplement.

---

## 5. Sources

- Clef model card: https://huggingface.co/Cloudflare/clef · Clef-flash: https://huggingface.co/Cloudflare/clef-flash
- Clef deep dive (limits, Workers AI pricing): https://flaviocopes.com/clef/ · https://www.eesel.ai/blog/cloudflare-clef-pricing
- Jev deep dive (limits, signup pause): https://flaviocopes.com/jev/ · TypeSafe docs: https://docs.typesafe.ai/concepts/system-one · OpenRouter: https://openrouter.ai/models/typesafe/jev-1.13
- Jev general availability: https://explainx.ai/blog/jev-general-availability-no-waitlist-2026
- autotrust JEV-9B: https://huggingface.co/autotrust/JEV-9B
- OpenAI Decisions API: https://www.orcarouter.ai/blog/openai-decisions-api-gpt-6-luna · https://propakistani.pk/2026/10/07/openai-launches-decisions-api-with-up-to-10x-faster-responses/
- Jev Decision Index: https://aiweekly.co/alerts/jev-decision-index-ranks-30-open-weight-decision-models
- toolrank: https://pypi.org/project/toolrank/ · https://huggingface.co/yasinyaman/toolrank-emb-8b
- ToolRet: https://arxiv.org/abs/2503.01763 · https://github.com/mangopy/tool-retrieval-benchmark · https://huggingface.co/datasets/mangopy/ToolRet-Queries
- LiveMCPBench: https://github.com/icip-cas/LiveMCPBench · https://arxiv.org/abs/2508.01780
- MCP-Bench: https://github.com/Accenture/mcp-bench
- MCP-Zero: https://github.com/xfey/MCP-Zero
- AgentDojo: https://github.com/ethz-spylab/agentdojo
- SCOUT: https://arxiv.org/abs/2608.23992 · Prune 'n Predict: https://arxiv.org/abs/2501.00555
- LangChain middleware: https://docs.langchain.com/oss/python/langchain/middleware/custom · https://reference.langchain.com/python/langchain/agents/middleware/tool_selection/LLMToolSelectorMiddleware
- Strands hooks and tool registry: https://strandsagents.com/docs/api/python/strands.hooks.events/ · https://strandsagents.com/docs/api/python/strands.tools.registry/
- TorchCP: https://jmlr.org/papers/v26/24-2141.html
- Free GPU quotas: https://gpuperhour.com/blog/free-cloud-gpus-and-credits
