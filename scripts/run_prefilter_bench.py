"""Stage-1 prefilter benchmark: recall@K of candidate prefilters on ToolRet (no API calls).

For every encoder, tool-document variant and query variant, ranks each step's catalog with
BM25, dense retrieval and their RRF fusion (the current prefilter), and reports recall@K
(all gold tools in the top K), which caps end-to-end coverage (design §8.4).

  - encoders: any sentence-transformers model; `none` runs BM25 only
  - --docs toolret: the original ToolRet tool documents
  - --docs expanded: the same tools with LLM-written profiles from Tool-REX / Tool-DE
    (arXiv 2510.22670; function, when to use, tags), generated from the tool docs alone
  - --queries query: the step as given; query+instruction adds ToolRet's per-step
    instruction, which GPT-4o wrote while seeing the gold tools, so it is an oracle
    upper bound, not a deployable input
  - --scopes pooled (whole catalog), routed<m> (tools of the m best-matching toolsets first),
    toolset (the step's own toolset first: the tools its agent is connected to)
  - --match exact (gold tool ids) and/or duplicate (a tool with the gold tool's description,
    from the same toolset, also counts; RoTBench ships ~38 renamed copies of each tool)
  - an encoder spec `A+B` fuses BM25 and several encoders with RRF (an ensemble)

  python scripts/run_prefilter_bench.py --tag full --encoders sentence-transformers/multi-qa-MiniLM-L6-cos-v1
  python scripts/run_prefilter_bench.py --tag small --categories code,customized \
      --encoders sentence-transformers/multi-qa-MiniLM-L6-cos-v1,LiquidAI/LFM2.5-Embedding-350M
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cte.data import SUBSET_TOOLSET, SUBSETS, expand_tool_docs, load_toolret, toolset_of
from cte.metrics import paired_bootstrap
from cte.pipeline import write_json
from cte.report import md_table
from cte.retrieval import ROUTE_BONUS, Encoder, HybridPrefilter, route_bonus

KS = (10, 20, 50, 100, 200)
BASELINE = ("hybrid", "toolret", "query", "sentence-transformers/multi-qa-MiniLM-L6-cos-v1", "pooled", "exact")


def load_instructions(root: Path) -> dict[str, str]:
    out = {}
    for s in SUBSETS:
        df = pd.read_parquet(root / "queries" / f"{s}.parquet", columns=["id", "instruction"])
        out.update(zip(df["id"], df["instruction"].fillna("")))
    return out


def worst_gold_rank(scores: np.ndarray, classes: list[np.ndarray], unretrievable: np.ndarray | None = None) -> float:
    """0-based rank of the lowest-ranked gold tool, where each gold tool counts as found when any
    tool of its class (itself, or its duplicates) is found: all gold tools are in the top K iff < K."""
    if unretrievable is not None and any(unretrievable[c].all() for c in classes):
        return np.inf
    g = np.array([scores[c].max() for c in classes])
    return float((scores[None, :] > g[:, None]).sum(1).max())


def duplicate_classes(catalog) -> list[np.ndarray]:
    """For each tool, the tools of the same toolset with the same description (itself included)."""
    norm = lambda t: " ".join(t.description.lower().split())
    groups: dict[tuple, list[int]] = {}
    for j, t in enumerate(catalog.tools):
        groups.setdefault((toolset_of(t.id), norm(t)), []).append(j)
    return [np.array(groups[(toolset_of(t.id), norm(t))]) for t in catalog.tools]


def lfm2_compat() -> None:
    """LFM2.5-Embedding's remote code (bidirectional patch) predates transformers 5 passing `seq_idx`
    to the short-conv layers; drop that argument (it is None for unpacked batches)."""
    from transformers.models.lfm2.modeling_lfm2 import Lfm2ShortConv

    slow = Lfm2ShortConv.slow_forward
    if not getattr(slow, "_drops_seq_idx", False):
        def patched(self, *a, seq_idx=None, **kw):
            return slow(self, *a, **kw)
        patched._drops_seq_idx = True
        Lfm2ShortConv.slow_forward = patched


def scope_bonus(scores: np.ndarray, scope: str, toolset_idx: np.ndarray, step_ts: np.ndarray) -> np.ndarray | int:
    """`pooled`: the whole catalog; `routed<m>`: tools of the m best-matching toolsets first;
    `toolset`: the step's own toolset first (oracle: the agent's connected servers)."""
    if scope == "pooled":
        return 0
    if scope == "toolset":
        return ROUTE_BONUS * (toolset_idx[None, :] == step_ts[:, None])
    return route_bonus(scores, toolset_idx, int(scope.removeprefix("routed")))


def query_latency_ms(enc: Encoder, texts: list[str]) -> float:
    """CPU latency of encoding one query (batch of 16, uncached), in milliseconds."""
    t0 = time.perf_counter()
    enc.model.encode(texts, batch_size=16, prompt_name=enc.query_prompt, normalize_embeddings=True)
    return 1000 * (time.perf_counter() - t0) / len(texts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data/toolret"))
    p.add_argument("--cache", type=Path, default=Path("cache"))
    p.add_argument("--out", type=Path, default=Path("results/prefilter"))
    p.add_argument("--tag", default="full")
    p.add_argument("--categories", default="web,code,customized")
    p.add_argument("--encoders", default="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    p.add_argument("--docs", default="toolret,expanded")
    p.add_argument("--queries", default="query,query+instruction")
    p.add_argument("--max-chars", type=int, default=600)
    p.add_argument("--scopes", default="pooled",
                   help="comma list from pooled, routed1, routed2, routed3, toolset (oracle)")
    p.add_argument("--match", default="exact", help="comma list from exact, duplicate")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--trust-remote-code", default="",
                   help="comma list of encoders allowed to run code from their model repo (e.g. LFM2.5's "
                        "bidirectional-attention patch); review that code first")
    args = p.parse_args()
    out = args.out / args.tag
    out.mkdir(parents=True, exist_ok=True)

    catalogs, steps = load_toolret(args.data)
    cats = args.categories.split(",")
    catalogs = {c: catalogs[c] for c in cats}
    steps = [s for s in steps if s.category in catalogs]
    instructions = load_instructions(args.data)
    encoders = [e for e in args.encoders.split(",") if e != "none"]
    doc_variants = {"toolret": catalogs}
    if "expanded" in args.docs.split(","):
        doc_variants["expanded"] = expand_tool_docs(catalogs, args.data)
    query_variants = args.queries.split(",")
    single = np.array([len(s.gold) == 1 for s in steps])
    scopes, matches = args.scopes.split(","), args.match.split(",")
    category = np.array([s.category for s in steps])
    print(f"{len(steps)} steps ({single.sum()} single-tool) over {sum(len(c) for c in catalogs.values())} tools "
          f"in {cats}")

    ranks: dict[tuple, np.ndarray] = {}
    routed: dict[tuple, np.ndarray] = {}  # gold toolset among the routed toolsets
    latency, params = {}, {}
    prefilters = {(docs, cat): HybridPrefilter(catalog, None, max_chars=args.max_chars)  # BM25 index per catalog
                  for docs, cat_map in doc_variants.items() for cat, catalog in cat_map.items()}
    # Duplicates are judged on the original descriptions (expanded profiles differ between copies).
    dup = {cat: duplicate_classes(catalog) for cat, catalog in catalogs.items()}
    for enc_name in [None] + encoders:
        names = enc_name.split("+") if enc_name else []
        encs = []
        for n in names:
            remote = n in args.trust_remote_code.split(",")
            encs.append(Encoder(n, args.cache, trust_remote_code=remote))
            if remote and "LFM2" in n:
                lfm2_compat()
        if encs:
            params[enc_name] = sum(x.numel() for e in encs for x in e.model.parameters()) / 1e6
            latency[enc_name] = sum(query_latency_ms(e, [s.situation for s in steps[:64]]) for e in encs)
        label = enc_name or "-"
        for docs, cat_map in doc_variants.items():
            for qv in query_variants:
                t0 = time.time()
                methods = ["bm25"] if not encs else ["dense", "hybrid"]
                for key in [(m, docs, qv, label, sc, mt) for m in methods for sc in scopes for mt in matches]:
                    ranks[key] = np.full(len(steps), np.inf)
                    routed[key] = np.zeros(len(steps), bool)
                for cat, catalog in cat_map.items():
                    pf = prefilters[(docs, cat)]
                    embs = [e.encode([t.text(args.max_chars) for t in catalog.tools]) for e in encs]
                    rows = [i for i, s in enumerate(steps) if s.category == cat]
                    for b in range(0, len(rows), args.batch):
                        idx = rows[b:b + args.batch]
                        texts = [steps[i].situation if qv == "query"
                                 else f"{instructions.get(steps[i].id, '')} {steps[i].situation}".strip() for i in idx]
                        bm = pf._bm25_scores(texts)
                        step_ts = np.array([pf.toolsets.index(SUBSET_TOOLSET[steps[i].subset]) for i in idx])
                        if not encs:
                            scored = {"bm25": (bm, bm <= 0)}
                        else:
                            sims = [e.encode(texts, is_query=True) @ emb.T for e, emb in zip(encs, embs)]
                            if len(sims) == 1:
                                dense = sims[0]
                            else:  # ensemble: RRF over the encoders, ties broken by mean cosine
                                dense = sum(pf._ranks_to_rrf(x + 1.0) for x in sims) + 1e-6 * np.mean(sims, 0)
                            fused = (pf._ranks_to_rrf(bm) + sum(pf._ranks_to_rrf(x + 1.0) for x in sims)
                                     + 1e-6 * np.mean(sims, 0))
                            scored = {"dense": (dense, None), "hybrid": (fused, None)}
                        for m, (sc_mat, unretrievable) in scored.items():
                            for sc in scopes:
                                bonus = scope_bonus(sc_mat, sc, pf.toolset_idx, step_ts)
                                adj = np.asarray(sc_mat, np.float64) + bonus
                                for j, i in enumerate(idx):
                                    gold = [catalog.index[g] for g in steps[i].gold]
                                    hit_ts = (np.ndim(bonus) == 0) or bool(bonus[j, gold[0]] > 0)
                                    for mt in matches:
                                        classes = ([np.array([g]) for g in gold] if mt == "exact"
                                                   else [dup[cat][g] for g in gold])
                                        key = (m, docs, qv, label, sc, mt)
                                        ranks[key][i] = worst_gold_rank(
                                            adj[j], classes, None if unretrievable is None else unretrievable[j])
                                        routed[key][i] = hit_ts
                    for e in encs:
                        e.cache.save()
                for sc in scopes:
                    for mt in matches:
                        r = ranks[(methods[-1], docs, qv, label, sc, mt)]
                        print(f"{(enc_name or 'bm25')[:60]:60s} docs={docs:8s} q={qv:5s} {methods[-1]:6s} {sc:8s} {mt:9s} "
                              f"single R@50={np.mean(r[single] < 50):.3f} R@100={np.mean(r[single] < 100):.3f} "
                              f"R@200={np.mean(r[single] < 200):.3f} ({time.time() - t0:.0f}s)")
        del encs

    rows_json = []
    base = ranks.get(BASELINE)
    for (m, docs, qv, enc_name, sc, mt), r in ranks.items():
        delta = (paired_bootstrap((r < 50).astype(float), (base < 50).astype(float))
                 if base is not None and (m, docs, qv, enc_name, sc, mt) != BASELINE else None)
        row = {"method": m, "encoder": enc_name, "docs": docs, "query": qv, "scope": sc, "match": mt,
               "routing_accuracy": float(routed[(m, docs, qv, enc_name, sc, mt)].mean()),
               "delta_recall@50_vs_current": delta,
               **{f"recall@{k}": float(np.mean(r < k)) for k in KS},
               **{f"recall@{k}_single": float(np.mean(r[single] < k)) for k in KS},
               **{f"recall@{k}_multi": float(np.mean(r[~single] < k)) for k in KS if (~single).any()},
               **{f"recall@50[{c}]": float(np.mean(r[category == c] < 50)) for c in cats},
               "params_M": params.get(enc_name), "query_ms_cpu": latency.get(enc_name)}
        rows_json.append(row)
    np.savez(out / "ranks.npz", step_ids=np.array([s.id for s in steps]),
             **{"|".join(k): v for k, v in ranks.items()})
    write_json(out / "results.json", {"n_steps": len(steps), "n_single": int(single.sum()), "categories": cats,
                                      "n_tools": sum(len(c) for c in catalogs.values()), "rows": rows_json})
    write_report(rows_json, out, len(steps), int(single.sum()), cats, sum(len(c) for c in catalogs.values()))
    print(f"wrote {out / 'report.md'}")


def write_report(rows: list[dict], out: Path, n: int, n_single: int, cats: list[str], n_tools: int) -> None:
    short = lambda e: " + ".join(x.split("/")[-1] for x in e.split("+")) if e != "-" else "–"
    rows = sorted(rows, key=lambda r: (r["query"] != "query", -r["recall@50"]))
    ci = lambda d: f"{d['mean_diff']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]" if d else "–"
    head = lambda r: [r["method"], short(r["encoder"]), r["docs"], r["query"], r["scope"], r["match"]]
    table = [[*head(r), *(r[f"recall@{k}"] for k in KS), *(r[f"recall@{k}_single"] for k in (20, 50, 100)),
              r.get("recall@50_multi"), r["routing_accuracy"] if r["scope"].startswith("routed") else None,
              ci(r["delta_recall@50_vs_current"]), r["params_M"], r["query_ms_cpu"]] for r in rows]
    lines = [
        "# Stage-1 prefilter benchmark (ToolRet)",
        "",
        f"{n} steps ({n_single} single-tool) over {n_tools} tools, catalogs: {', '.join(cats)}. "
        "Recall@K = all gold tools among the top K. `hybrid` = BM25 + dense fused with RRF (the current "
        "prefilter design). `expanded` docs = Tool-REX tool profiles. `query+instruction` uses ToolRet's "
        "target-aware instruction (written with the gold tools in view): an oracle upper bound only. "
        "Scope: `pooled` ranks the whole catalog; `routed<m>` ranks the tools of the m toolsets whose best "
        "tool scores highest first (hierarchical server-then-tool routing, no labels used); `toolset` ranks "
        "the step's own toolset first (oracle: the agent's connected toolset is known). Routing accuracy = "
        "the gold toolset is among the m routed ones. Match `duplicate`: a tool from the same toolset with the "
        "gold tool's exact description also counts (RoTBench ships renamed copies of each tool). An encoder "
        "`A + B` fuses BM25 and both encoders with RRF. "
        "ΔR@50 = paired bootstrap difference against the current prefilter (hybrid, MiniLM, ToolRet docs, "
        "pooled). Latency = CPU time to encode one query (4 cores, fp32, batch 16).",
        "",
        md_table(["method", "encoder", "docs", "query", "scope", "match", *(f"R@{k}" for k in KS), "R@20 single",
                  "R@50 single", "R@100 single", "R@50 multi", "routing acc.", "ΔR@50 vs current [95% CI]",
                  "params (M)", "query ms (CPU)"], table),
        "",
        "## Recall@50 by catalog",
        "",
        md_table(["method", "encoder", "docs", "query", "scope", "match", *cats],
                 [[*head(r), *(r[f"recall@50[{c}]"] for c in cats)] for r in rows]),
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
