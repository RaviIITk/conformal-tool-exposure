"""Phase 2 proxy (RQ2/H2): agent success per token under each tool-exposure condition.

Needs OPENROUTER_API_KEY (decision model = Mercury Decide) and a chat model for the agent
(CTE_AGENT_MODEL via OPENAI_API_KEY, or an OpenRouter model id via OPENROUTER_API_KEY).
Calls: one decision call per test step plus one agent call per step per condition.

Conditions (design Section 10.3):
  B-k  fixed top-k from the hybrid prefilter (k = 1, 3, 5)
  D    decision-model top-1
  E    decision model + conformal set (proposed; empty set -> top-1)
  F    E capped at k_max
  G    E with online ACI updating the threshold from feedback
Condition A (all tools) is infeasible at ToolRet catalog sizes; C (agent-driven
discovery) needs the full multi-step loop and is left to the live-benchmark phase.

Example:
  python scripts/run_phase2_proxy.py --max-steps 1500 --n-steps 300
"""

from __future__ import annotations

import argparse

import numpy as np
from dotenv import load_dotenv

from cte import conformal as cp
from cte.agent_eval import AgentProxy
from cte.pipeline import add_common_args, build_setup, make_source, subset_by, write_json
from cte.report import md_table


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--decision-source", default="mercury-choice")
    p.add_argument("--n-steps", type=int, default=300)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--k-max", type=int, default=5)
    p.add_argument("--gamma", type=float, default=0.01)
    args = p.parse_args()
    out = args.out / "phase2"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    S = build_setup(args)
    src = make_source(args.decision_source, args)
    src.fit(S.dev)
    single = subset_by(S.eval, S.eval.n_gold == 1)
    perm = rng.permutation(len(single))
    cal = single.take(np.sort(perm[: len(perm) // 2]))
    test = single.take(np.sort(perm[len(perm) // 2:][: args.n_steps]))

    # Threshold calibrated on steps whose gold survived the prefilter (Stage-2 guarantee).
    cal_p = src.probs(cal)
    hit = cal.hit_all
    cal_scores = cp.lac_scores(cal_p[hit], cal.gold_mask[hit], cal.n_gold[hit])
    qhat = cp.conformal_quantile(cal_scores, args.alpha)
    probs = src.probs(test)
    k = test.k

    conformal = cp.with_top1_fallback(cp.lac_sets(probs, qhat), probs)
    aci = cp.AdaptiveConformal(cal_scores, args.alpha, args.gamma)
    aci_sets = np.zeros_like(conformal)
    true_scores = cp.lac_scores(probs, test.gold_mask, test.n_gold)
    for i in range(len(test)):
        aci_sets[i] = cp.with_top1_fallback(cp.lac_sets(probs[i:i + 1], aci.qhat()), probs[i:i + 1])[0]
        if np.isfinite(true_scores[i]):  # feedback is only available when the gold tool was shown
            aci.update(true_scores[i])

    def first_k(n):
        m = np.zeros((len(test), k), bool)
        m[:, :n] = True
        return m

    top1 = np.zeros_like(conformal)
    top1[np.arange(len(test)), probs.argmax(1)] = True
    conditions = {"B-1": first_k(1), "B-3": first_k(3), "B-5": first_k(5), "D (decision top-1)": top1,
                  "E (conformal)": conformal, f"F (conformal, cap {args.k_max})": cp.cap_sets(conformal, probs, args.k_max),
                  "G (conformal + ACI)": aci_sets}

    agent = AgentProxy(args.cache)
    rows, results = [], {"qhat": qhat, "alpha": args.alpha, "n_test": len(test), "agent_model": agent.model,
                         "decision_source": args.decision_source, "conditions": {}}
    for name, sets in conditions.items():
        tool_lists, gold_pos = [], []
        for i in range(len(test)):
            members = np.flatnonzero(sets[i])
            members = members[np.argsort(-probs[i, members])] if not name.startswith("B") else members
            tool_lists.append([test.tools[i][j] for j in members])
            gold_pos.append({pos for pos, j in enumerate(members) if test.gold_mask[i, j]})
        recs = agent.run([s.situation for s in test.steps], tool_lists, desc=name)
        success = np.array([r["chosen"] is not None and r["chosen"] in g for r, g in zip(recs, gold_pos)])
        tokens = np.array([r["prompt_tokens"] for r in recs])
        res = {"success": float(success.mean()), "gold_shown": float(np.mean([bool(g) for g in gold_pos])),
               "avg_tools": float(sets.sum(1).mean()), "prompt_tokens": float(tokens.mean()),
               "success_per_1k_tokens": float(success.sum() / tokens.sum() * 1000)}
        results["conditions"][name] = res
        rows.append([name, res["avg_tools"], res["gold_shown"], res["success"], res["prompt_tokens"],
                     res["success_per_1k_tokens"]])
        print(f"{name:28s} tools={res['avg_tools']:.2f} shown={res['gold_shown']:.3f} "
              f"success={res['success']:.3f} tokens={res['prompt_tokens']:.0f}")
    write_json(out / "results.json", results)
    (out / "report.md").write_text(
        f"# Phase 2 proxy results (RQ2): agent `{agent.model}`, decision source `{args.decision_source}`\n\n"
        f"{len(test)} single-tool test steps, α = {args.alpha}, q̂ = {qhat:.4f}.\n\n"
        + md_table(["condition", "avg tools shown", "gold tool shown", "agent success", "prompt tokens",
                    "successes per 1k tokens"], rows) + "\n")


if __name__ == "__main__":
    main()
