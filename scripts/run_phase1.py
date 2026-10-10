"""Phase 1 (RQ1/H1): offline coverage and set size for each Stage-2 score source.

Examples:
  # free, local sources on every ToolRet step
  python scripts/run_phase1.py
  # add the OpenAI decision model on a 1,500-step stratified sample (needs OPENAI_API_KEY)
  python scripts/run_phase1.py --max-steps 1500 \
      --sources bm25,dense,hybrid,ce,ce-pred,decisions-choice,decisions-predicate,logprob
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from dotenv import load_dotenv

from cte import conformal as cp
from cte.experiment import repeated_cp
from cte.metrics import calibration_error, holm, paired_bootstrap, reliability_curve, top1_accuracy
from cte.openai_scorers import DECISIONS_PRICE_PER_M_INPUT
from cte.pipeline import add_common_args, build_setup, make_source, subset_by, write_json
from cte.report import coverage_hist, line_plot, md_table, reliability_plot

ALPHAS = (0.05, 0.10, 0.20)
H1_ALPHA = 0.10
MIN_VIEW_STEPS = 10


def views(pool):
    """Step subsets evaluated in Phase 1.

    e2e:    all steps; a gold tool missed by the prefilter counts as non-coverage (Section 8.4).
    stage2: steps whose gold tools all survived the prefilter; isolates the score source.
    """
    single = pool.n_gold == 1
    return {
        "single_e2e": single,
        "single_stage2": single & pool.hit_all,
        "multi_e2e": ~single,
        "multi_stage2": ~single & pool.hit_all,
    }


def evaluate_source(src, probs, pool, args) -> tuple[dict, dict]:
    res = {"mode": src.mode, "top1_accuracy": top1_accuracy(probs, pool.gold_mask),
           "ece": calibration_error(probs, pool.gold_mask, src.mode), "cp": {}}
    step_sizes = {}
    groups = np.array([s.category for s in pool.steps])
    for view, mask in views(pool).items():
        if mask.sum() < MIN_VIEW_STEPS:
            continue
        sub, sub_p = subset_by(pool, mask), probs[mask]
        methods = ["lac", "aps"] if (view.startswith("single") and src.mode == "choice") else ["lac"]
        if view.startswith("multi"):
            methods = ["crc", "lac"]
        for method in methods:
            for alpha in ALPHAS:
                r = repeated_cp(sub_p, sub, method, alpha, args.splits, k_max=args.k_max, seed=args.seed,
                                groups=groups[mask])
                step_sizes[(view, method, alpha)] = r.pop("step_avg_size")
                res["cp"][f"{view}|{method}|{alpha}"] = r
    return res, step_sizes


def deployment_thresholds(probs, pool) -> dict:
    """Thresholds calibrated on every eval step, for use at runtime (e.g. by the middleware)."""
    v = views(pool)
    s, m = v["single_stage2"], v["multi_stage2"]
    out = {}
    for a in ALPHAS:
        if s.sum() >= MIN_VIEW_STEPS:
            out[f"lac|{a}"] = cp.conformal_quantile(cp.lac_scores(probs[s], pool.gold_mask[s], pool.n_gold[s]), a)
        if m.sum() >= MIN_VIEW_STEPS:
            out[f"crc|{a}"] = cp.crc_threshold(probs[m], pool.gold_mask[m], pool.n_gold[m], a)
    return out


def h1_tests(step_sizes: dict, names: list[str], reference: str) -> list[dict]:
    """H1: at matched target coverage, compare each source's set size with the reference."""
    key = ("single_stage2", "lac", H1_ALPHA)
    rows = []
    for name in names:
        if name == reference:
            continue
        r = paired_bootstrap(step_sizes[name][key], step_sizes[reference][key], seed=0)
        rows.append({"source": name, "reference": reference, **r})
    for row, p in zip(rows, holm([r["p"] for r in rows])):
        row["p_holm"] = p
    return rows


def write_report(results: dict, h1: list[dict], out) -> None:
    names = list(results["sources"])
    lines = [
        "# Phase 1 results: coverage and set size by score source (RQ1)",
        "",
        f"Steps: {results['n_dev']} dev (K choice, calibration fitting) + {results['n_eval']} eval "
        f"(repeated {results['splits']} random 50/50 calibration/test splits). "
        f"Prefilter: BM25 + `{results['encoder']}` with RRF, K = {results['k']}.",
        "",
        "## Prefilter recall on the dev split (hard ceiling on end-to-end coverage)",
        "",
        md_table(["K", "recall (all gold tools in top-K)"],
                 [[k, v] for k, v in sorted(results["recall_dev"].items(), key=lambda kv: int(kv[0]))]),
        "",
        "## Score-source quality on eval steps",
        "",
        md_table(["source", "mode", "top-1 accuracy", "ECE"],
                 [[n, r["mode"], r["top1_accuracy"], r["ece"]] for n, r in results["sources"].items()]),
    ]
    for view, title, method in [
        ("single_stage2", "Single-tool steps, Stage 2 only (gold tool survived the prefilter), LAC", "lac"),
        ("single_e2e", "Single-tool steps, end to end (prefilter misses count as failures), LAC", "lac"),
        ("single_stage2", "Single-tool steps, Stage 2 only, APS (choice sources)", "aps"),
        ("multi_stage2", "Multi-tool steps, Stage 2 only, conformal risk control (false-negative rate ≤ α)", "crc"),
        ("multi_e2e", "Multi-tool steps, end to end, conformal risk control", "crc"),
    ]:
        rows = []
        for n in names:
            for a in ALPHAS:
                r = results["sources"][n]["cp"].get(f"{view}|{method}|{a}")
                if r is None:
                    continue
                rows.append([n, a, r["n"], r["coverage"], r["coverage_p05"], r["theory"]["p05"], r["fnr"],
                             r["avg_size"], r["singleton_rate"], r["empty_rate"],
                             r[f"coverage_capped{results['k_max']}"], r["threshold_finite"]])
        if rows:
            lines += ["", f"## {title}", "",
                      md_table(["source", "α", "n", "coverage (mean)", "coverage (5th pct)", "theory 5th pct",
                                "FNR", "avg set size", "singleton", "empty", f"coverage capped k={results['k_max']}",
                                "threshold finite"], rows)]
    lines += ["", f"## H1: set size at matched target coverage (single-tool, Stage 2, LAC, α = {H1_ALPHA})", "",
              "Paired bootstrap over steps of the difference in average set size (negative = smaller sets than the "
              "reference), Holm-corrected.", "",
              md_table(["source", "reference", "mean size diff", "95% CI low", "95% CI high", "p (Holm)"],
                       [[r["source"], r["reference"], r["mean_diff"], r["ci_low"], r["ci_high"], r["p_holm"]]
                        for r in h1])]
    cat_rows = []
    for n in names:
        r = results["sources"][n]["cp"].get(f"single_stage2|lac|{H1_ALPHA}", {})
        cat_rows.append([n] + [r.get(f"coverage[{c}]") for c in ("web", "code", "customized")])
    lines += ["", f"## Conditional coverage by catalog (single-tool, Stage 2, LAC, α = {H1_ALPHA})", "",
              md_table(["source", "web", "code", "customized"], cat_rows)]
    usage = {n: r["usage"] for n, r in results["sources"].items() if "usage" in r}
    if usage:
        lines += ["", "## API usage", "",
                  md_table(["source", "calls", "input tokens", "cost (USD)", "median latency (s)", "p95 latency (s)",
                            "refusals"],
                           [[n, u["calls"], u["input_tokens"], u.get("cost_usd"), u["latency_median_s"],
                             u["latency_p95_s"], u["refusals"]] for n, u in usage.items()])]
    lines += ["", "Figures: `set_size_vs_alpha.png`, `coverage_vs_target.png`, `reliability.png`, "
                  "`coverage_distribution.png`."]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--splits", type=int, default=100)
    p.add_argument("--k-max", type=int, default=5)
    args = p.parse_args()
    out = args.out / "phase1"
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    S = build_setup(args)
    print(f"setup {time.time() - t0:.0f}s | K={S.k} | dev recall {S.recalls} | dev={len(S.dev)} eval={len(S.eval)}")

    names = args.sources.split(",")
    results = {"k": S.k, "recall_dev": S.recalls, "n_dev": len(S.dev), "n_eval": len(S.eval),
               "splits": args.splits, "k_max": args.k_max, "encoder": args.encoder, "sources": {}}
    all_probs, step_sizes, curves = {}, {}, {}
    for name in names:
        t0 = time.time()
        src = make_source(name, args)
        src.fit(S.dev)
        probs = src.probs(S.eval)
        all_probs[name] = probs
        res, step_sizes[name] = evaluate_source(src, probs, S.eval, args)
        if hasattr(src, "usage"):
            res["usage"] = src.usage(S.eval)
            price = DECISIONS_PRICE_PER_M_INPUT if name.startswith("decisions") else 0.0  # :free models cost $0
            res["usage"]["cost_usd"] = res["usage"]["input_tokens"] * price / 1e6
        if src.mode == "choice":
            top = probs.argmax(1)
            rows = np.arange(len(probs))
            curves[name] = reliability_curve(probs[rows, top], S.eval.gold_mask[rows, top].astype(float))
        res["params"] = list(getattr(src, "params", ()))
        res["thresholds"] = deployment_thresholds(probs, S.eval)
        results["sources"][name] = res
        r = res["cp"][f"single_stage2|lac|{H1_ALPHA}"]
        print(f"{name:20s} {time.time() - t0:6.0f}s  top1={res['top1_accuracy']:.3f} ece={res['ece']:.3f} "
              f"| stage2 LAC α=0.1: cov={r['coverage']:.3f} size={r['avg_size']:.2f}")

    reference = "dense" if "dense" in names else names[0]
    h1 = h1_tests(step_sizes, names, reference)
    for extra in ("decisions-choice", "decisions-predicate"):
        if extra in names:
            h1 += [dict(r, comparison=f"vs {extra}") for r in h1_tests(step_sizes, names, extra)]
    np.savez(out / "probs.npz", step_ids=np.array([s.id for s in S.eval.steps]), **all_probs)
    write_json(out / "results.json", {**results, "h1": h1})

    for view in ("single_stage2", "single_e2e"):
        line_plot(out / f"set_size_vs_alpha_{view}.png" if view != "single_stage2" else out / "set_size_vs_alpha.png",
                  {n: (list(ALPHAS), [results["sources"][n]["cp"][f"{view}|lac|{a}"]["avg_size"] for a in ALPHAS])
                   for n in names},
                  f"Average set size ({view}, LAC)", "α", "avg set size")
    line_plot(out / "coverage_vs_target.png",
              {f"{n} ({v.split('_')[1]})": ([1 - a for a in ALPHAS],
                                            [results["sources"][n]["cp"][f"{v}|lac|{a}"]["coverage"] for a in ALPHAS])
               for n in names for v in ("single_stage2", "single_e2e")},
              "Empirical vs target coverage (single-tool steps, LAC)", "target 1-α", "empirical coverage",
              diagonal=True)
    if curves:
        reliability_plot(out / "reliability.png", curves)
    r = results["sources"][names[-1]]["cp"][f"single_stage2|lac|{H1_ALPHA}"]
    coverage_hist(out / "coverage_distribution.png", r["coverage_by_split"], r["theory"]["p05"], 1 - H1_ALPHA,
                  f"{names[-1]}: coverage over {args.splits} random splits (α={H1_ALPHA})")
    write_report(results, h1, out)
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
