"""End to end with the deployment prefilter: Mercury Decide on the agent's own toolset.

1. Steps: single-tool ToolRet steps of --categories. A stratified sample of --n steps goes to
   Mercury; every other step chooses K (the smallest K whose recall reaches --target + --margin).
2. Prefilter: the toolset of the step's own dataset (the tools its agent is connected to) is
   ranked by an RRF ensemble of --encoder over expanded tool docs; the top K go to Stage 2.
3. Mercury Decide (Mode A: one choice over the K candidates) scores each sampled step.
4. Conformal sets (LAC), leave-one-out: each step's threshold is calibrated on all other
   sampled steps (n - 1 points), so alpha down to about 1/n gives a finite threshold.
   A prefilter miss counts as a failure (score +inf), so coverage is end to end.

  python scripts/run_deploy_mercury.py --n 50
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from cte import conformal as cp
from cte.data import expand_tool_docs, load_toolret, toolset_of
from cte.experiment import repeated_cp, stratified_sample
from cte.metrics import calibration_error, split_coverage_theory
from cte.openai_scorers import SystemOneSource
from cte.pipeline import write_json
from cte.report import md_table
from cte.retrieval import Encoder, HybridPrefilter, build_pool

ALPHAS = (0.02, 0.05, 0.10)


def duplicate_mask(pool, originals: dict) -> np.ndarray:
    """Candidates that are the gold tool or a same-description copy of it from the same toolset."""
    key = lambda tid: (toolset_of(tid), " ".join(originals[tid].description.lower().split()))
    return np.array([[key(t.id) in {key(g) for g in st.gold} for t in tools]
                     for st, tools in zip(pool.steps, pool.tools)])


def best_copy(probs: np.ndarray, dup: np.ndarray) -> np.ndarray:
    """Gold mask that keeps, per step, the copy Mercury ranks highest (a set covers some copy iff it covers it)."""
    out = np.zeros_like(dup)
    rows = np.flatnonzero(dup.any(1))
    out[rows, np.where(dup[rows], probs[rows], -1.0).argmax(1)] = True
    return out


def loo_lac(probs: np.ndarray, gold_mask: np.ndarray, n_gold: np.ndarray, alpha: float) -> np.ndarray:
    """Leave-one-out split conformal: step i's set uses the threshold from the other n - 1 steps."""
    scores = cp.lac_scores(probs, gold_mask, n_gold)
    sets = np.zeros_like(gold_mask)
    for i in range(len(probs)):
        q = cp.conformal_quantile(np.delete(scores, i), alpha)
        sets[i] = cp.lac_sets(probs[i:i + 1], q)[0]
    return sets


def hit_at(probs: np.ndarray, gold_mask: np.ndarray, k: int) -> float:
    top = np.argsort(-probs, axis=1, kind="stable")[:, :k]
    return float(np.take_along_axis(gold_mask, top, 1).any(1).mean())


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data/toolret"))
    p.add_argument("--cache", type=Path, default=Path("cache"))
    p.add_argument("--out", type=Path, default=Path("results/deploy_mercury"))
    p.add_argument("--categories", default="code,customized")
    p.add_argument("--encoder", default="Lux1997/Tool-Embed-0.6B+mangopy/ToolRet-trained-bge-base-en-v1.5")
    p.add_argument("--use-bm25", action="store_true", help="also fuse BM25 (default: encoders only)")
    p.add_argument("--max-chars", type=int, default=600)
    p.add_argument("--n", type=int, default=50, help="steps sent to Mercury (one call each)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target", type=float, default=0.95)
    p.add_argument("--margin", type=float, default=0.02)
    p.add_argument("--k", type=int, default=None, help="override the K chosen on the other steps")
    p.add_argument("--k-grid", default="50,100,150,200,250,300")
    p.add_argument("--splits", type=int, default=200)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    catalogs, steps = load_toolret(args.data)
    cats = args.categories.split(",")
    originals = {t.id: t for c in cats for t in catalogs[c].tools}
    catalogs = expand_tool_docs({c: catalogs[c] for c in cats}, args.data)
    steps = [s for s in steps if s.category in cats and len(s.gold) == 1]
    sample = stratified_sample(steps, args.n, args.seed)
    sample_ids = {s.id for s in sample}
    others = [s for s in steps if s.id not in sample_ids]
    print(f"{len(steps)} single-tool steps in {cats}: {len(sample)} to Mercury, {len(others)} choose K")

    t0 = time.time()
    encoders = [Encoder(n, args.cache) for n in args.encoder.split("+")]
    prefilters = {c: HybridPrefilter(cat, encoders, max_chars=args.max_chars, scope="toolset",
                                     use_bm25=args.use_bm25) for c, cat in catalogs.items()}
    grid = sorted(int(k) for k in args.k_grid.split(","))
    k_max = max(grid + [args.k or 0])
    pool_others = build_pool(others, prefilters, k_max)
    pool = build_pool(sample, prefilters, k_max)
    recall_others = {k: float(pool_others.gold_mask[:, :k].any(1).mean()) for k in grid}
    k = args.k or next((k for k in grid if recall_others[k] >= args.target + args.margin), grid[-1])
    pool = pool.truncate(k)
    print(f"prefilter {time.time() - t0:.0f}s | recall on the other steps {recall_others} | K = {k} | "
          f"recall on the Mercury sample {pool.hit_all.mean():.3f}")

    src = SystemOneSource("choice", args.cache)
    probs = src.probs(pool)
    usage = src.usage(pool)
    dup = duplicate_mask(pool, originals)
    n, ng = len(pool), pool.n_gold
    res = {"categories": cats, "encoder": args.encoder, "use_bm25": args.use_bm25, "n": n, "k": k,
           "target": args.target, "margin": args.margin, "recall_other_steps": recall_others,
           "n_other_steps": len(others), "prefilter_recall": float(pool.hit_all.mean()),
           "prefilter_recall_duplicate": float(dup.any(1).mean()), "usage": usage,
           "ece": calibration_error(probs, pool.gold_mask, "choice"), "mercury": {}, "conformal": {}}
    for name, gm in (("exact", pool.gold_mask), ("duplicate", best_copy(probs, dup))):
        reached = gm.any(1)
        res["mercury"][name] = {
            **{f"hit@{h}": hit_at(probs, gm, h) for h in (1, 3, 5, 10)},
            **{f"hit@{h}_given_prefilter": hit_at(probs[reached], gm[reached], h) if reached.any() else None
               for h in (1, 3, 5)}}
        for a in ALPHAS:
            sets = loo_lac(probs, gm, ng, a)
            covered = (sets & gm).any(1)
            rep = repeated_cp(probs, replace(pool, gold_mask=gm), "lac", a, args.splits, k_max=5, seed=args.seed)
            res["conformal"][f"{name}|{a}"] = {
                "loo_coverage": float(covered.mean()), "loo_avg_size": float(sets.sum(1).mean()),
                "loo_median_size": float(np.median(sets.sum(1))), "loo_singleton": float((sets.sum(1) == 1).mean()),
                "loo_empty": float((sets.sum(1) == 0).mean()), "loo_theory": split_coverage_theory(n - 1, a),
                "split_coverage": rep["coverage"], "split_coverage_p05": rep["coverage_p05"],
                "split_avg_size": rep["avg_size"], "split_threshold_finite": rep["threshold_finite"]}
    write_json(args.out / "results.json", res)
    write_report(res, args.out)
    for a in ALPHAS:
        r = res["conformal"][f"exact|{a}"]
        print(f"alpha={a}: end-to-end coverage {r['loo_coverage']:.3f}, avg set size {r['loo_avg_size']:.2f}")
    print(f"wrote {args.out / 'report.md'}")


def write_report(res: dict, out: Path) -> None:
    m = res["mercury"]
    lines = [
        "# End to end: deployment prefilter + Mercury Decide",
        "",
        f"{res['n']} single-tool steps from ToolRet {', '.join(res['categories'])} (stratified sample), one Mercury "
        f"Decide call each. Prefilter: the step's own toolset, ranked by an RRF ensemble of "
        f"`{res['encoder']}`{' and BM25' if res['use_bm25'] else ''} over expanded tool docs. K = {res['k']}, the "
        f"smallest K whose recall on the other {res['n_other_steps']} steps reaches {res['target']} + {res['margin']}.",
        "",
        "## Prefilter",
        "",
        md_table(["K", "recall on the other steps"], [[k, v] for k, v in sorted(res["recall_other_steps"].items(),
                                                                                key=lambda kv: int(kv[0]))]),
        "",
        f"On the Mercury sample at K = {res['k']}: recall {res['prefilter_recall']:.3f} (exact tool id), "
        f"{res['prefilter_recall_duplicate']:.3f} (a same-description copy also counts).",
        "",
        "## Mercury Decide: right tool among its top k (end to end)",
        "",
        md_table(["match", "hit@1", "hit@3", "hit@5", "hit@10", "hit@1 | prefilter hit", "hit@5 | prefilter hit"],
                 [[name, r["hit@1"], r["hit@3"], r["hit@5"], r["hit@10"], r["hit@1_given_prefilter"],
                   r["hit@5_given_prefilter"]] for name, r in m.items()]),
        "",
        f"ECE of the top choice: {res['ece']:.3f}.",
        "",
        "## Conformal tool sets (LAC), end to end",
        "",
        "Leave-one-out: each step's threshold is calibrated on the other n − 1 steps. `split` = 200 random 50/50 "
        "calibration/test splits, for comparison (needs n/2 calibration points, so small α gives no finite "
        "threshold). Theory 5th pct = Beta-distribution lower band for n − 1 calibration points.",
        "",
        md_table(["match", "α", "target", "coverage (LOO)", "theory 5th pct", "avg set size", "median size",
                  "singleton", "empty", "coverage (split)", "split threshold finite"],
                 [[k.split("|")[0], float(k.split("|")[1]), 1 - float(k.split("|")[1]), r["loo_coverage"],
                   r["loo_theory"]["p05"], r["loo_avg_size"], r["loo_median_size"], r["loo_singleton"], r["loo_empty"],
                   r["split_coverage"], r["split_threshold_finite"]] for k, r in res["conformal"].items()]),
        "",
        "## API usage",
        "",
        md_table(["calls", "input tokens", "median latency (s)", "p95 latency (s)", "refusals"],
                 [[res["usage"]["calls"], res["usage"]["input_tokens"], res["usage"]["latency_median_s"],
                   res["usage"]["latency_p95_s"], res["usage"]["refusals"]]]),
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
