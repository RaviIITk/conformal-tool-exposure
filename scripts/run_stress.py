"""Phase 3 (RQ3/H3, RQ4/H4): coverage under catalog drift, distribution shift and poisoned descriptions.

Calibration always uses the clean catalog; perturbations hit the test steps afterwards,
as they would in deployment. Single-tool steps, LAC, Stage-2 view by default (the
threshold is calibrated on steps whose gold tool survived the prefilter).

Examples:
  python scripts/run_stress.py --source ce
  python scripts/run_stress.py --source decisions-choice --max-steps 1500   # needs OPENAI_API_KEY
"""

from __future__ import annotations

import argparse

import numpy as np
from dotenv import load_dotenv

from cte import conformal as cp
from cte.data import Catalog
from cte.metrics import set_metrics
from cte.pipeline import add_common_args, build_setup, make_source, subset_by, write_json
from cte.report import line_plot, md_table
from cte.retrieval import HybridPrefilter, Pool, build_pool
from cte.stress import drift, poison, sanitize_catalog


def stage2(pool: Pool, probs: np.ndarray, view: str) -> tuple[Pool, np.ndarray]:
    if view == "e2e":
        return pool, probs
    keep = pool.hit_all
    return subset_by(pool, keep), probs[keep]


def score_on(catalogs: dict[str, Catalog], steps, S, src, k: int) -> tuple[Pool, np.ndarray]:
    prefilters = {c: HybridPrefilter(cat, S.encoder) for c, cat in catalogs.items()}
    pool = build_pool(steps, prefilters, k)
    return pool, src.probs(pool)


def stream(qhat_static: float, cal_scores: np.ndarray, test_scores: np.ndarray, alpha: float, gamma: float):
    """Replay the test stream: static split-CP threshold vs adaptive conformal inference."""
    aci = cp.AdaptiveConformal(cal_scores, alpha, gamma)
    static_err = test_scores > qhat_static
    aci_err, aci_q = [], []
    for s in test_scores:
        aci_q.append(aci.qhat())
        aci_err.append(aci.update(s))
    return static_err, np.array(aci_err), np.array(aci_q)


def rolling(err: np.ndarray, w: int) -> np.ndarray:
    cov = 1 - err.astype(float)
    return np.convolve(cov, np.ones(w) / w, mode="valid")


def recovery_steps(err: np.ndarray, target: float, w: int, tol: float = 0.02) -> int | None:
    r = rolling(err, w)
    ok = np.flatnonzero(r >= target - tol)
    return int(ok[0] + w) if len(ok) else None


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--source", default="ce")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.01, help="ACI step size")
    p.add_argument("--view", choices=["stage2", "e2e"], default="stage2")
    p.add_argument("--poison-fracs", default="0.01,0.05")
    p.add_argument("--window", type=int, default=200, help="rolling-coverage window for plots")
    args = p.parse_args()
    out = args.out / "stress"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    S = build_setup(args)
    src = make_source(args.source, args)
    src.fit(S.dev)
    single = subset_by(S.eval, S.eval.n_gold == 1)
    perm = rng.permutation(len(single))
    cal_pool, test_pool = single.take(np.sort(perm[: len(perm) // 2])), single.take(np.sort(perm[len(perm) // 2:]))
    cal_p = src.probs(cal_pool)
    cal_v, cal_vp = stage2(cal_pool, cal_p, args.view)
    cal_scores = cp.lac_scores(cal_vp, cal_v.gold_mask, cal_v.n_gold)
    qhat = cp.conformal_quantile(cal_scores, args.alpha)
    print(f"{args.source}: calibrated on {len(cal_v)} steps, qhat={qhat:.4f}")
    results = {"source": args.source, "alpha": args.alpha, "view": args.view, "qhat": qhat,
               "n_cal": len(cal_v), "k": S.k}

    def evaluate(pool: Pool, probs: np.ndarray, extra_ids: set[str] = frozenset(), label: str = "") -> dict:
        v, vp = stage2(pool, probs, args.view)
        sets = cp.lac_sets(vp, qhat)
        m = set_metrics(sets, v.gold_mask, v.n_gold)
        m["n"] = len(v)
        m["prefilter_recall"] = float(pool.hit_all.mean())
        if extra_ids:
            ids = [[t.id in extra_ids for t in row] for row in v.tools]
            in_k = np.array(ids, dtype=bool)
            m["flagged_in_prefilter"] = float(in_k.any(1).mean())
            m["flagged_in_set"] = float((in_k & sets).any(1).mean())
            m["flagged_prob_mass"] = float(np.where(in_k, vp, 0).sum(1).mean())
        print(f"  {label:32s} cov={m['coverage']:.3f} size={m['avg_size']:.2f} n={m['n']}"
              + (f" flagged_in_set={m['flagged_in_set']:.3f}" if extra_ids else ""))
        return m

    test_steps = test_pool.steps
    test_p = src.probs(test_pool)
    results["clean"] = evaluate(test_pool, test_p, label="clean test")

    # ---- RQ3a: catalog drift --------------------------------------------------------------
    gold_test = {g for s in test_steps for g in s.gold}
    drifted = {}
    for c, cat in S.catalogs.items():
        others = [t for oc, ocat in S.catalogs.items() if oc != c for t in ocat.tools]
        drifted[c] = drift(cat, rng, gold_test, others)
    d_pool, d_p = score_on(drifted, test_steps, S, src, S.k)
    dup_ids = {t.id for cat in drifted.values() for t in cat.tools if t.id.startswith("dup_")}
    results["drift_static"] = evaluate(d_pool, d_p, dup_ids, "drift, static split CP")

    v, vp = stage2(d_pool, d_p, args.view)
    order = rng.permutation(len(v))
    test_scores = cp.lac_scores(vp, v.gold_mask, v.n_gold)[order]
    vp_stream = vp[order]
    static_err, aci_err, aci_q = stream(qhat, cal_scores, test_scores, args.alpha, args.gamma)
    half = len(order) // 2
    oracle_q = cp.conformal_quantile(test_scores[:half], args.alpha)
    results["drift_stream"] = {
        "static_coverage": float(1 - static_err.mean()),
        "aci_coverage": float(1 - aci_err.mean()),
        "aci_coverage_second_half": float(1 - aci_err[half:].mean()),
        "oracle_recalibrated_coverage_second_half": float((test_scores[half:] <= oracle_q).mean()),
        "aci_final_qhat": float(aci_q[-1]), "oracle_qhat": oracle_q,
        "aci_recovery_steps": recovery_steps(aci_err, 1 - args.alpha, args.window),
        "aci_avg_size": float(((1 - vp_stream) <= aci_q[:, None]).sum(1).mean()),
    }
    print(f"  drift stream: static={results['drift_stream']['static_coverage']:.3f} "
          f"ACI={results['drift_stream']['aci_coverage']:.3f}")
    line_plot(out / "drift_rolling_coverage.png",
              {"static split CP": (np.arange(len(rolling(static_err, args.window))), rolling(static_err, args.window)),
               f"online ACI (γ={args.gamma})": (np.arange(len(rolling(aci_err, args.window))),
                                                rolling(aci_err, args.window))},
              f"Rolling coverage after catalog drift ({args.source})", "test step", f"coverage (window {args.window})",
              hline=1 - args.alpha)

    # ---- RQ3b: distribution shift between catalogs ----------------------------------------
    cats = np.array([s.category for s in single.steps])
    sp = src.probs(single)
    sv_mask = single.hit_all if args.view == "stage2" else np.ones(len(single), bool)
    shift = {}
    for cal_cat in ("web", "code"):
        cal_m, test_m = (cats == cal_cat) & sv_mask, (cats != cal_cat) & sv_mask
        cs = cp.lac_scores(sp[cal_m], single.gold_mask[cal_m], single.n_gold[cal_m])
        ts = cp.lac_scores(sp[test_m], single.gold_mask[test_m], single.n_gold[test_m])
        ts = ts[rng.permutation(len(ts))]
        q = cp.conformal_quantile(cs, args.alpha)
        st_err, a_err, _ = stream(q, cs, ts, args.alpha, args.gamma)
        shift[f"calibrate_{cal_cat}"] = {"n_cal": int(cal_m.sum()), "n_test": int(test_m.sum()), "qhat": q,
                                         "static_coverage": float(1 - st_err.mean()),
                                         "aci_coverage": float(1 - a_err.mean())}
        print(f"  shift cal={cal_cat}: static={1 - st_err.mean():.3f} ACI={1 - a_err.mean():.3f}")
    results["shift"] = shift

    # ---- RQ4: adversarial descriptions + description-sanity defence ----------------------
    results["adversarial"] = {}
    for frac in [float(x) for x in args.poison_fracs.split(",")]:
        poisoned, sanitized, bad = {}, {}, set()
        for c, cat in S.catalogs.items():
            poisoned[c], ids = poison(cat, frac, rng, gold_test)
            bad |= ids
        clean_baseline = evaluate(test_pool, test_p, bad, f"clean, same tools ({frac:.0%})")
        p_pool, p_p = score_on(poisoned, test_steps, S, src, S.k)
        attacked = evaluate(p_pool, p_p, bad, f"poisoned {frac:.0%}")
        flagged = set()
        for c, cat in poisoned.items():
            sanitized[c], f = sanitize_catalog(cat)
            flagged |= f
        s_pool, s_p = score_on(sanitized, test_steps, S, src, S.k)
        defended = evaluate(s_pool, s_p, bad, f"poisoned {frac:.0%} + sanity check")
        n_tools = sum(len(c) for c in poisoned.values())
        results["adversarial"][str(frac)] = {
            "n_poisoned": len(bad), "clean_baseline": clean_baseline, "attacked": attacked, "defended": defended,
            "detector_precision": len(flagged & bad) / max(1, len(flagged)),
            "detector_recall": len(flagged & bad) / max(1, len(bad)),
            "detector_false_positive_rate": len(flagged - bad) / max(1, n_tools - len(bad)),
        }

    write_json(out / f"results_{args.source}.json", results)
    write_report(results, out, args)


def write_report(r: dict, out, args) -> None:
    def row(name, m):
        return [name, m["n"], m["prefilter_recall"], m["coverage"], m["avg_size"], m.get("flagged_in_set"),
                m.get("flagged_prob_mass")]

    hdr = ["condition", "n", "prefilter recall", "coverage", "avg set size", "flagged tool in set",
           "flagged prob. mass"]
    ds = r["drift_stream"]
    lines = [
        f"# Phase 3 results: stress tests ({r['source']}, α = {r['alpha']}, {r['view']} view, K = {r['k']})",
        "",
        f"Threshold q̂ = {r['qhat']:.4f}, calibrated on {r['n_cal']} clean single-tool steps.",
        "",
        "## RQ3a: catalog drift (remove 20% of tools, add 20% from other catalogs, rename 20%, "
        "near-duplicate lookalikes for half of the gold tools)",
        "",
        md_table(hdr, [row("clean test", r["clean"]), row("drifted, static q̂ (flagged = lookalikes)",
                                                          r["drift_static"])]),
        "",
        md_table(["stream metric", "value"], [[k, v] for k, v in ds.items()]),
        "",
        "## RQ3b: distribution shift (calibrate on one catalog, test on the others)",
        "",
        md_table(["calibration", "n cal", "n test", "q̂", "static coverage", "ACI coverage"],
                 [[k, v["n_cal"], v["n_test"], v["qhat"], v["static_coverage"], v["aci_coverage"]]
                  for k, v in r["shift"].items()]),
        "",
        "## RQ4: poisoned tool descriptions (flagged = poisoned tools)",
        "",
    ]
    for frac, a in r["adversarial"].items():
        lines += [f"### {float(frac):.0%} of tools poisoned ({a['n_poisoned']} tools)", "",
                  md_table(hdr, [row("clean catalog (same tools, unpoisoned)", a["clean_baseline"]),
                                 row("poisoned", a["attacked"]), row("poisoned + sanity check", a["defended"])]),
                  "",
                  f"Sanity-check detector: precision {a['detector_precision']:.3f}, recall {a['detector_recall']:.3f}, "
                  f"false-positive rate on clean tools {a['detector_false_positive_rate']:.4f}.", ""]
    lines += ["Figure: `drift_rolling_coverage.png`."]
    (out / f"report_{r['source']}.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / ('report_' + r['source'] + '.md')}")


if __name__ == "__main__":
    main()
