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

from cte.data import SUBSETS, expand_tool_docs, load_toolret
from cte.metrics import paired_bootstrap
from cte.pipeline import write_json
from cte.report import md_table
from cte.retrieval import Encoder, HybridPrefilter

KS = (10, 20, 50, 100)
BASELINE = ("hybrid", "toolret", "query", "sentence-transformers/multi-qa-MiniLM-L6-cos-v1")  # current prefilter


def load_instructions(root: Path) -> dict[str, str]:
    out = {}
    for s in SUBSETS:
        df = pd.read_parquet(root / "queries" / f"{s}.parquet", columns=["id", "instruction"])
        out.update(zip(df["id"], df["instruction"].fillna("")))
    return out


def worst_gold_rank(scores: np.ndarray, gold: np.ndarray, unretrievable: np.ndarray | None = None) -> float:
    """0-based rank of the lowest-ranked gold tool (all gold tools are in the top K iff it is < K)."""
    g = scores[gold]
    if unretrievable is not None and unretrievable[gold].any():
        return np.inf
    return float((scores[None, :] > g[:, None]).sum(1).max())


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
    category = np.array([s.category for s in steps])
    print(f"{len(steps)} steps ({single.sum()} single-tool) over {sum(len(c) for c in catalogs.values())} tools "
          f"in {cats}")

    ranks: dict[tuple, np.ndarray] = {}
    latency, params = {}, {}
    prefilters = {(docs, cat): HybridPrefilter(catalog, None, max_chars=args.max_chars)  # BM25 index per catalog
                  for docs, cat_map in doc_variants.items() for cat, catalog in cat_map.items()}
    for enc_name in [None] + encoders:
        remote = enc_name in args.trust_remote_code.split(",")
        enc = Encoder(enc_name, args.cache, trust_remote_code=remote) if enc_name else None
        if remote and "LFM2" in enc_name:
            lfm2_compat()
        if enc:
            params[enc_name] = sum(x.numel() for x in enc.model.parameters()) / 1e6
            latency[enc_name] = query_latency_ms(enc, [s.situation for s in steps[:64]])
        for docs, cat_map in doc_variants.items():
            for qv in query_variants:
                t0 = time.time()
                methods = ["bm25"] if enc is None else ["dense", "hybrid"]
                for m in methods:
                    ranks[(m, docs, qv, enc_name or "-")] = np.full(len(steps), np.inf)
                for cat, catalog in cat_map.items():
                    pf = prefilters[(docs, cat)]
                    emb = enc.encode([t.text(args.max_chars) for t in catalog.tools]) if enc else None
                    rows = [i for i, s in enumerate(steps) if s.category == cat]
                    for b in range(0, len(rows), args.batch):
                        idx = rows[b:b + args.batch]
                        texts = [steps[i].situation if qv == "query"
                                 else f"{instructions.get(steps[i].id, '')} {steps[i].situation}".strip() for i in idx]
                        bm = pf._bm25_scores(texts)
                        if enc is None:
                            for j, i in enumerate(idx):
                                gold = np.array([catalog.index[g] for g in steps[i].gold])
                                ranks[("bm25", docs, qv, "-")][i] = worst_gold_rank(bm[j], gold, bm[j] <= 0)
                            continue
                        dense = enc.encode(texts, is_query=True) @ emb.T
                        fused = pf._ranks_to_rrf(bm) + pf._ranks_to_rrf(dense + 1.0) + 1e-6 * dense
                        for j, i in enumerate(idx):
                            gold = np.array([catalog.index[g] for g in steps[i].gold])
                            ranks[("dense", docs, qv, enc_name)][i] = worst_gold_rank(dense[j], gold)
                            ranks[("hybrid", docs, qv, enc_name)][i] = worst_gold_rank(fused[j], gold)
                    if enc:
                        enc.cache.save()
                r = ranks[(methods[-1], docs, qv, enc_name or "-")]
                print(f"{(enc_name or 'bm25'):45s} docs={docs:8s} q={qv:17s} {methods[-1]} "
                      f"R@50={np.mean(r < 50):.3f} R@100={np.mean(r < 100):.3f} ({time.time() - t0:.0f}s)")
        del enc

    rows_json = []
    base = ranks.get(BASELINE)
    for (m, docs, qv, enc_name), r in ranks.items():
        delta = (paired_bootstrap((r < 50).astype(float), (base < 50).astype(float))
                 if base is not None and (m, docs, qv, enc_name) != BASELINE else None)
        row = {"method": m, "encoder": enc_name, "docs": docs, "query": qv, "delta_recall@50_vs_current": delta,
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
    short = lambda e: e.split("/")[-1] if e != "-" else "–"
    rows = sorted(rows, key=lambda r: (r["query"] != "query", -r["recall@50"]))
    ci = lambda d: f"{d['mean_diff']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]" if d else "–"
    table = [[r["method"], short(r["encoder"]), r["docs"], r["query"], *(r[f"recall@{k}"] for k in KS),
              r["recall@50_single"], r.get("recall@50_multi"), ci(r["delta_recall@50_vs_current"]),
              r["params_M"], r["query_ms_cpu"]] for r in rows]
    lines = [
        "# Stage-1 prefilter benchmark (ToolRet)",
        "",
        f"{n} steps ({n_single} single-tool) over {n_tools} tools, catalogs: {', '.join(cats)}. "
        "Recall@K = all gold tools among the top K. `hybrid` = BM25 + dense fused with RRF (the current "
        "prefilter design). `expanded` docs = Tool-REX tool profiles. `query+instruction` uses ToolRet's "
        "target-aware instruction (written with the gold tools in view): an oracle upper bound only. "
        "ΔR@50 = paired bootstrap difference against the current prefilter (hybrid, MiniLM, ToolRet docs). "
        "Latency = CPU time to encode one query (4 cores, fp32, batch 16).",
        "",
        md_table(["method", "encoder", "docs", "query", *(f"R@{k}" for k in KS), "R@50 single", "R@50 multi",
                  "ΔR@50 vs current [95% CI]", "params (M)", "query ms (CPU)"], table),
        "",
        "## Recall@50 by catalog",
        "",
        md_table(["method", "encoder", "docs", "query", *cats],
                 [[r["method"], short(r["encoder"]), r["docs"], r["query"], *(r[f"recall@50[{c}]"] for c in cats)]
                  for r in rows]),
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
