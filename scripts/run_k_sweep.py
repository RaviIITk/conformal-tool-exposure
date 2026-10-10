"""Prefilter K sweep: Stage-1 recall and final conformal results at each prefilter K.

Phase 1 fixes the prefilter at K = 50. This script runs the same steps at every K in
--k-values and reports, for each K:
  - prefilter results: recall (all gold tools among the K candidates) on dev and eval steps;
  - final results per score source: top-1 accuracy, hit@k, and LAC coverage and set size,
    Stage 2 only and end to end, plus API usage.

Sources that call an API are run from the smallest K up; if the API stops answering, the
sweep keeps every K finished so far and reports which ones it skipped.

  python scripts/run_k_sweep.py --max-steps 59 --single-tool-only \
      --encoder sentence-transformers/multi-qa-MiniLM-L6-cos-v1 \
      --k-values 10,20,30,50 --sources dense,mercury-choice
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from dotenv import load_dotenv

from cte.pipeline import add_common_args, build_setup, make_source, write_json
from cte.report import line_plot, md_table
from hit_at_k import hit_at_k
from run_phase1 import ALPHAS, H1_ALPHA, evaluate_source

HIT_KS = (1, 3, 5)


def prefilter_stats(dev, ev) -> dict:
    single = ev.n_gold == 1
    return {"recall_dev": float(dev.hit_all.mean()), "recall_eval": float(ev.hit_all.mean()),
            "recall_eval_single": float(ev.hit_all[single].mean()) if single.any() else None,
            "recall_eval_multi": float(ev.hit_all[~single].mean()) if (~single).any() else None,
            "n_dev": len(dev), "n_eval": len(ev), "n_eval_single": int(single.sum())}


def run_source(src, dev, ev, args) -> dict:
    src.fit(dev)
    probs = src.probs(ev)
    res, _ = evaluate_source(src, probs, ev, args)
    single = ev.n_gold == 1
    hits = hit_at_k(probs[single], ev.gold_mask[single])
    res["hit_at_k_e2e"] = {k: hits[k] for k in HIT_KS}
    if hasattr(src, "usage"):
        res["usage"] = src.usage(ev)
    return res


def cp_row(res: dict, view: str, alpha: float) -> dict:
    return res["cp"].get(f"{view}|lac|{alpha}", {})


def write_report(results: dict, out) -> None:
    ks = sorted(results["by_k"], key=int)
    names = results["sources"]
    lines = [
        "# Prefilter K sweep: prefilter and final results by K",
        "",
        f"Steps: {results['n_dev']} dev + {results['n_eval']} eval "
        f"({results['splits']} random 50/50 calibration/test splits per K). "
        f"Prefilter: BM25 + `{results['encoder']}` with RRF, top K kept. Score sources: {', '.join(names)}.",
        "",
        "## Prefilter results (recall = all gold tools among the K candidates)",
        "",
        md_table(["K", "dev recall", "eval recall (all)", "eval recall (single-tool)", "eval recall (multi-tool)"],
                 [[int(k), p["recall_dev"], p["recall_eval"], p["recall_eval_single"], p["recall_eval_multi"]]
                  for k in ks for p in [results["by_k"][k]["prefilter"]]]),
        "",
        f"## Final results, single-tool steps, LAC, α = {H1_ALPHA}",
        "",
        "Stage 2 = steps whose gold tool survived the prefilter; end to end = all single-tool steps, "
        "prefilter misses count as failures. hit@k is end to end.",
        "",
    ]
    rows, skipped = [], []
    for n in names:
        for k in ks:
            res = results["by_k"][k]["sources"].get(n)
            if res is None:
                skipped.append(f"{n} at K = {k}")
                continue
            s2, e2e = cp_row(res, "single_stage2", H1_ALPHA), cp_row(res, "single_e2e", H1_ALPHA)
            rows.append([n, int(k), res["top1_accuracy"], *(res["hit_at_k_e2e"][h] for h in HIT_KS),
                         s2.get("n"), s2.get("coverage"), s2.get("avg_size"),
                         e2e.get("coverage"), e2e.get("avg_size")])
    lines.append(md_table(["source", "K", "top-1 acc.", *(f"hit@{h}" for h in HIT_KS), "Stage-2 n",
                           "Stage-2 coverage", "Stage-2 avg set size", "e2e coverage", "e2e avg set size"], rows))
    full = []
    for n in names:
        for k in ks:
            res = results["by_k"][k]["sources"].get(n)
            if res is None:
                continue
            for view in ("single_stage2", "single_e2e"):
                for a in ALPHAS:
                    r = cp_row(res, view, a)
                    if r:
                        full.append([n, int(k), view.split("_")[1], a, r["n"], r["coverage"], r["coverage_p05"],
                                     r["avg_size"], r["singleton_rate"], r[f"coverage_capped{results['k_max']}"],
                                     r["threshold_finite"]])
    lines += ["", "## Final results, all α (single-tool steps, LAC)", "",
              md_table(["source", "K", "view", "α", "n", "coverage (mean)", "coverage (5th pct)", "avg set size",
                        "singleton", f"coverage capped k={results['k_max']}", "threshold finite"], full)]
    usage = [[n, int(k), u["calls"], u["input_tokens"], u["latency_median_s"], u["latency_p95_s"], u["refusals"]]
             for n in names for k in ks
             for u in [(results["by_k"][k]["sources"].get(n) or {}).get("usage")] if u]
    if usage:
        lines += ["", "## API usage", "",
                  md_table(["source", "K", "calls", "input tokens", "median latency (s)", "p95 latency (s)",
                            "refusals"], usage)]
    if skipped:
        lines += ["", "Not run (the API stopped answering): " + "; ".join(skipped) + "."]
    lines += ["", "Figures: `recall_vs_k.png`, `coverage_vs_k.png`, `set_size_vs_k.png`."]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.set_defaults(sources="dense,mercury-choice")
    p.add_argument("--k-values", default="10,20,30,50", help="comma list of prefilter K values")
    p.add_argument("--splits", type=int, default=100)
    p.add_argument("--k-max", type=int, default=5)
    args = p.parse_args()
    ks = sorted(int(k) for k in args.k_values.split(","))
    args.k, args.k_grid = ks[-1], args.k_values
    out = args.out / "k_sweep"
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    S = build_setup(args)  # candidates for the largest K; each K below is a prefix of that ranking
    print(f"setup {time.time() - t0:.0f}s | K values {ks} | dev={len(S.dev)} eval={len(S.eval)}")

    names = args.sources.split(",")
    sources = {n: make_source(n, args) for n in names}
    results = {"k_values": ks, "sources": names, "n_dev": len(S.dev), "n_eval": len(S.eval), "splits": args.splits,
               "k_max": args.k_max, "encoder": args.encoder, "by_k": {}}
    stopped = set()
    for k in ks:
        dev, ev = S.dev.truncate(k), S.eval.truncate(k)
        entry = {"prefilter": prefilter_stats(dev, ev), "sources": {}}
        print(f"K={k:4d} prefilter recall dev={entry['prefilter']['recall_dev']:.3f} "
              f"eval={entry['prefilter']['recall_eval']:.3f}")
        for n, src in sources.items():
            if n in stopped:
                continue
            t0 = time.time()
            try:
                res = run_source(src, dev, ev, args)
            except RuntimeError as e:
                print(f"  {n}: stopped at K={k} ({e}); keeping the K values already finished")
                stopped.add(n)
                continue
            entry["sources"][n] = res
            s2, e2e = cp_row(res, "single_stage2", H1_ALPHA), cp_row(res, "single_e2e", H1_ALPHA)
            print(f"  {n:16s} {time.time() - t0:5.0f}s top1={res['top1_accuracy']:.3f} | α=0.1 stage2 "
                  f"cov={s2.get('coverage', float('nan')):.3f} size={s2.get('avg_size', float('nan')):.2f} | "
                  f"e2e cov={e2e.get('coverage', float('nan')):.3f} size={e2e.get('avg_size', float('nan')):.2f}")
        results["by_k"][str(k)] = entry
        write_json(out / "results.json", results)

    done = [str(k) for k in ks]
    line_plot(out / "recall_vs_k.png",
              {"dev": (ks, [results["by_k"][k]["prefilter"]["recall_dev"] for k in done]),
               "eval": (ks, [results["by_k"][k]["prefilter"]["recall_eval"] for k in done])},
              "Prefilter recall vs K", "prefilter K", "recall")
    for metric, fname, title in (("coverage", "coverage_vs_k.png", "Coverage vs K"),
                                 ("avg_size", "set_size_vs_k.png", "Average set size vs K")):
        series = {}
        for n in names:
            for view in ("single_stage2", "single_e2e"):
                pts = [(int(k), cp_row(results["by_k"][k]["sources"][n], view, H1_ALPHA)[metric])
                       for k in done if n in results["by_k"][k]["sources"]]
                if pts:
                    series[f"{n} ({view.split('_')[1]})"] = tuple(map(list, zip(*pts)))
        if series:
            line_plot(out / fname, series, f"{title} (single-tool, LAC, α={H1_ALPHA})", "prefilter K", metric,
                      hline=1 - H1_ALPHA if metric == "coverage" else None)
    write_report(results, out)
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
