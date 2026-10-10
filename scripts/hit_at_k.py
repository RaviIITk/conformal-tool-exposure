"""Hit@k: if the system shows the agent k tools, how often is the right tool among them?

Reads only cached scores (no API calls). Writes results/hit_at_k.json.

  python scripts/hit_at_k.py
"""

from __future__ import annotations

import json
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from cte.pipeline import build_setup, make_source, write_json

KS = (1, 2, 3, 5, 10, 20, 50)


def hit_at_k(probs: np.ndarray, gold_mask: np.ndarray) -> dict:
    """Share of steps whose gold tool is in the top-k by score (misses by the prefilter count as failures)."""
    order = np.argsort(-probs, axis=1, kind="stable")
    gold_rank = np.where(np.take_along_axis(gold_mask, order, 1).any(1),
                         np.take_along_axis(gold_mask, order, 1).argmax(1), 10**6)
    return {k: float((gold_rank < k).mean()) for k in KS}


def args_for(**kw):
    a = types.SimpleNamespace(data=Path("data/toolret"), cache=Path("cache"),
                              encoder="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
                              ce_model="cross-encoder/ms-marco-MiniLM-L6-v2", max_steps=None, single_tool_only=False,
                              k=50, k_grid="20,50,100", dev_frac=0.2, seed=0)
    a.__dict__.update(kw)
    return a


def evaluate(args, names: list[str], blend: bool) -> dict:
    S = build_setup(args)
    pool = S.eval.take(np.flatnonzero(S.eval.n_gold == 1))
    probs = {}
    for n in names:
        src = make_source(n, args)
        src.fit(S.dev)
        probs[n] = src.probs(pool)
    probs["prefilter order"] = -np.arange(pool.k)[None, :].repeat(len(pool), 0).astype(float)
    if blend:
        probs["mercury+ce"] = 0.5 * probs["mercury-choice"] + 0.5 * probs["ce"]
    hit = pool.hit_all
    out = {"n_steps": len(pool), "recall_at_50": float(hit.mean()), "end_to_end": {}, "given_prefilter_hit": {},
           "by_catalog": {}, "by_benchmark": {}}
    for n, p in probs.items():
        out["end_to_end"][n] = hit_at_k(p, pool.gold_mask)
        out["given_prefilter_hit"][n] = hit_at_k(p[hit], pool.gold_mask[hit])
    for field, key in (("category", "by_catalog"), ("subset", "by_benchmark")):
        groups = defaultdict(list)
        for i, s in enumerate(pool.steps):
            groups[getattr(s, field)].append(i)
        for g, rows in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            rows = np.array(rows)
            out[key][g] = {"n": len(rows), "recall_at_50": float(hit[rows].mean()),
                           **{n: hit_at_k(p[rows], pool.gold_mask[rows]) for n, p in probs.items()}}
    return out


def main() -> None:
    load_dotenv(".env")
    full = evaluate(args_for(), ["bm25", "dense", "hybrid", "ce"], blend=False)
    sample = evaluate(args_for(max_steps=59, single_tool_only=True), ["hybrid", "ce", "mercury-choice"], blend=True)
    write_json(Path("results/hit_at_k.json"), {"ks": KS, "full_toolret": full, "mercury_sample": sample})
    for name, r in (("FULL ToolRet", full), ("MERCURY sample", sample)):
        print(f"\n{name}: {r['n_steps']} single-tool steps, recall@50 = {r['recall_at_50']:.3f}")
        for view in ("end_to_end", "given_prefilter_hit"):
            print(f"  [{view}]  k = " + "  ".join(f"{k:>5d}" for k in KS))
            for n, h in r[view].items():
                print(f"    {n:16s} " + "  ".join(f"{h[k]:5.3f}" for k in KS))


if __name__ == "__main__":
    main()
