"""Stage-1 prefilter at small K: can the right tool stay in the top 50? (no API calls)

Starts from the deployment prefilter of RESULTS.md section 7: the agent's own toolset ranked by
an RRF ensemble of Tool-Embed-0.6B and ToolRet-trained bge-base over Tool-REX expanded docs
(single-tool recall 0.90 at K = 50, 0.97 at K = 200). Tests, on ToolRet code + customized
single-tool steps:

  - more tool-trained encoders in the ensemble (`--extra-encoders`, ToolRet-trained bge-large,
    e5-large, ...; arXiv 2503.01763)
  - usage-based retrieval: score each tool by how similar the request is to past requests that
    used it (query-to-query kNN, as in Tool2Vec, arXiv 2409.02141), fused with RRF. Past
    requests come from the other cross-fitting folds of the same toolset; requests with the
    same text are left out, so a step never retrieves itself
  - cross-encoder reranking of the top `--rerank-depth` candidates (retrieve, then rerank) on a
    stratified sample of `--rerank-n` steps
  - adaptive K: instead of one K for everyone, a candidate count per toolset (Mondrian
    conformal) or per request (keep tools whose score is within a calibrated margin of the
    best one), calibrated for `--target` coverage on the other folds

Recall@K = the gold tool is among the top K of its own toolset. Ranks are taken within the
toolset, which is what the `toolset` scope puts first.

  python scripts/run_prefilter_k50.py
  python scripts/run_prefilter_k50.py --rerankers cross-encoder/ms-marco-MiniLM-L6-v2 --rerank-n 120
"""

from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from cte.cache import ArrayCache, text_key
from cte.data import SUBSET_TOOLSET, expand_tool_docs, load_toolret, toolset_of
from cte.experiment import stratified_sample
from cte.metrics import paired_bootstrap
from cte.pipeline import write_json
from cte.report import md_table
from cte.retrieval import Encoder

KS = (10, 20, 30, 50, 100, 200)
RRF_K = 60


def desc_rank(x: np.ndarray) -> np.ndarray:
    """0-based position of each entry when sorted best-first (ties keep their order)."""
    r = np.empty(len(x), int)
    r[np.argsort(-x, kind="stable")] = np.arange(len(x))
    return r


def rrf(x: np.ndarray) -> np.ndarray:
    return 1.0 / (RRF_K + 1 + desc_rank(x))


def gold_rank(s: np.ndarray, g: int) -> int:
    """0-based rank of the gold tool: the gold tool is in the top K iff this is < K."""
    return int((s > s[g]).sum())


def conformal_q(scores: np.ndarray, target: float) -> float:
    """Split-conformal quantile: ceil((n + 1) * target)-th smallest score, inf if n is too small."""
    n = len(scores)
    i = math.ceil((n + 1) * target) - 1
    return float(np.sort(scores)[i]) if i < n else math.inf


def smallest_k(ranks: np.ndarray, target: float) -> int:
    """Smallest K whose recall reaches the target (in-sample, descriptive)."""
    return int(np.sort(ranks)[math.ceil(target * len(ranks)) - 1]) + 1


class Setup:
    """Single-tool steps with their own toolset's tools (candidate indices) and gold position."""

    def __init__(self, args):
        catalogs, steps = load_toolret(args.data)
        cats = args.categories.split(",")
        self.catalogs = expand_tool_docs({c: catalogs[c] for c in cats}, args.data)
        self.all_steps = [s for s in steps if s.category in cats]
        self.steps = [s for s in self.all_steps if len(s.gold) == 1]
        members = {}
        for c, cat in self.catalogs.items():
            names = np.array([toolset_of(t.id) for t in cat.tools])
            for n in set(names):
                members[(c, n)] = np.flatnonzero(names == n)
        self.toolset = np.array([SUBSET_TOOLSET[s.subset] for s in self.steps])
        self.cand = [members[(s.category, SUBSET_TOOLSET[s.subset])] for s in self.steps]
        self.gold = np.array([int(np.flatnonzero(c == self.catalogs[s.category].index[s.gold[0]])[0])
                              for s, c in zip(self.steps, self.cand)])
        self.size = np.array([len(c) for c in self.cand])
        self.max_chars = args.max_chars
        # Cross-fitting folds over all steps (multi-tool ones only serve as past requests).
        perm = np.random.default_rng(args.seed).permutation(len(self.all_steps))
        fold_of = {self.all_steps[j].id: i % args.folds for i, j in enumerate(perm)}
        self.fold = np.array([fold_of[s.id] for s in self.steps])
        self.all_fold = np.array([fold_of[s.id] for s in self.all_steps])

    def texts(self, c: str) -> list[str]:
        return [t.text(self.max_chars) for t in self.catalogs[c].tools]


def encoder_sims(setup: Setup, name: str, cache: Path):
    """Per step: cosine to each tool of its toolset; query embeddings of eval and all steps; tool embeddings."""
    t0 = time.time()
    enc = Encoder(name, cache)
    tool_emb = {c: enc.encode(setup.texts(c)) for c in setup.catalogs}
    q_all = enc.encode([s.situation for s in setup.all_steps], is_query=True)
    enc.cache.save()
    pos = {s.id: i for i, s in enumerate(setup.all_steps)}
    q = q_all[[pos[s.id] for s in setup.steps]]
    sims = [tool_emb[s.category][c] @ q[i] for i, (s, c) in enumerate(zip(setup.steps, setup.cand))]
    print(f"{name}: {time.time() - t0:.0f}s", flush=True)
    return sims, q, q_all, tool_emb


def finetune_sims(setup: Setup, name: str, q_all: np.ndarray, tool_emb: dict, cache: Path, seed: int,
                  epochs: int = 1, batch: int = 16, lr: float = 2e-5, temperature: float = 0.05) -> list[np.ndarray]:
    """In-domain contrastive fine-tuning, cross-fitted: for each fold, train on the past requests of the
    other folds (request -> its gold tool, with in-batch negatives and one hard negative: a top-ranked
    other tool of the same toolset), then score the held-out fold's requests."""
    import torch
    from sentence_transformers import SentenceTransformer

    by_ts = {}
    for c, cat in setup.catalogs.items():
        names = np.array([toolset_of(t.id) for t in cat.tools])
        for n_ in set(names):
            by_ts[(c, n_)] = np.flatnonzero(names == n_)
    pos = {s.id: i for i, s in enumerate(setup.steps)}
    sims: list[np.ndarray | None] = [None] * len(setup.steps)
    rng = np.random.default_rng(seed)
    for f in np.unique(setup.fold):
        path = cache / f"ft_{name.replace('/', '__')}_fold{f}of{len(np.unique(setup.fold))}_seed{seed}.npz"
        test = [i for i in range(len(setup.steps)) if setup.fold[i] == f]
        if path.exists():
            z = np.load(path)
            tools_f, q_f = {c: z[f"tools_{c}"] for c in setup.catalogs}, z["queries"]
        else:
            t0 = time.time()
            examples = []
            for j, s in enumerate(setup.all_steps):
                if setup.all_fold[j] == f:
                    continue
                cat = setup.catalogs[s.category]
                members = by_ts[(s.category, SUBSET_TOOLSET[s.subset])]
                gold = [cat.index[g] for g in s.gold if g in cat.index]
                g = gold[rng.integers(len(gold))]
                order = members[np.argsort(-(tool_emb[s.category][members] @ q_all[j]))]
                negs = [t for t in order[:30] if t not in gold]
                neg = negs[rng.integers(len(negs))] if negs else None
                examples.append((s.situation, cat.tools[g].text(setup.max_chars),
                                 cat.tools[neg].text(setup.max_chars) if neg is not None else None))
            torch.manual_seed(seed)
            model = SentenceTransformer(name, device="cpu")
            model.max_seq_length = 256
            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=lr)
            emb = lambda texts: torch.nn.functional.normalize(
                model(model.tokenize(texts))["sentence_embedding"], dim=-1)
            for _ in range(epochs):
                order = rng.permutation(len(examples))
                for b in range(0, len(order), batch):
                    chunk = [examples[k] for k in order[b:b + batch]]
                    qs = emb([e[0] for e in chunk])
                    docs = emb([e[1] for e in chunk] + [e[2] for e in chunk if e[2] is not None])
                    loss = torch.nn.functional.cross_entropy(qs @ docs.T / temperature, torch.arange(len(chunk)))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            model.eval()
            with torch.no_grad():
                tools_f = {c: model.encode(setup.texts(c), batch_size=64, normalize_embeddings=True)
                           for c in setup.catalogs}
                q_f = model.encode([setup.steps[i].situation for i in test], batch_size=64, normalize_embeddings=True)
            np.savez(path, queries=q_f, **{f"tools_{c}": v for c, v in tools_f.items()})
            print(f"fine-tuned {name} fold {f}: {len(examples)} examples, {time.time() - t0:.0f}s", flush=True)
        for k, i in enumerate(test):
            s = setup.steps[i]
            sims[i] = tools_f[s.category][setup.cand[i]] @ q_f[k]
    return sims


def usage_scores(setup: Setup, q: list[np.ndarray], q_all: list[np.ndarray]) -> list[np.ndarray]:
    """Query-to-query kNN: a tool's score is the best similarity between this request and a past
    request (other folds, same toolset, different text) whose gold tools include it; tools never
    used before get -inf."""
    norm = lambda t: " ".join(t.lower().split())
    by_ts = defaultdict(list)
    for j, s in enumerate(setup.all_steps):
        by_ts[SUBSET_TOOLSET[s.subset]].append(j)
    out = []
    for i, (s, cand) in enumerate(zip(setup.steps, setup.cand)):
        cat = setup.catalogs[s.category]
        local = {int(t): k for k, t in enumerate(cand)}
        hist = [j for j in by_ts[SUBSET_TOOLSET[s.subset]]
                if setup.all_fold[j] != setup.fold[i] and norm(setup.all_steps[j].situation) != norm(s.situation)]
        score = np.full(len(cand), -np.inf)
        if hist:
            sim = np.mean([qa[hist] @ qe[i] for qe, qa in zip(q, q_all)], 0)
            for j, v in zip(hist, sim):
                for g in setup.all_steps[j].gold:
                    k = local.get(cat.index.get(g, -1))
                    if k is not None and v > score[k]:
                        score[k] = v
        out.append(score)
    return out


def rrf_partial(x: np.ndarray) -> np.ndarray:
    """RRF over the tools with a finite score; the rest get 0."""
    out = np.zeros(len(x))
    ok = np.isfinite(x)
    if ok.any():
        out[ok] = rrf(x[ok])
    return out


def gated_usage(setup: Setup, base: list[np.ndarray], usage: list[np.ndarray], k: int = 50,
                weights=(0.25, 0.5, 1.0), taus=(-np.inf, 0.5, 0.6, 0.7, 0.8, 0.9)):
    """base + w * RRF(usage kNN), counting only past requests at least tau similar. For each fold,
    (w, tau) maximise recall@k on the other folds, so the held-out fold never picks its own setting."""
    fused = lambda i, w, t: base[i] + w * rrf_partial(np.where(usage[i] >= t, usage[i], -np.inf))
    grid = {(w, t): np.array([gold_rank(fused(i, w, t), g) for i, g in enumerate(setup.gold)])
            for w in weights for t in taus}
    out, chosen = [None] * len(base), {}
    for f in np.unique(setup.fold):
        cal = setup.fold != f
        w, t = max(grid, key=lambda key: np.mean(grid[key][cal] < k))
        chosen[int(f)] = {"w": w, "tau": float(t)}
        for i in np.flatnonzero(~cal):
            out[i] = fused(i, w, t)
    return out, chosen


def rerank_scores(setup: Setup, rows: np.ndarray, base: list[np.ndarray], model_name: str, depth: int,
                  cache: Path, max_length: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Cross-encoder scores for each sampled step's top `depth` candidates under `base`."""
    from sentence_transformers import CrossEncoder

    store = ArrayCache(cache / f"rerank_{model_name.replace('/', '__')}.npz", 1)
    todo, out = [], {}
    for i in rows:
        top = np.argsort(-base[i], kind="stable")[:depth]
        s = setup.steps[i]
        docs = [setup.catalogs[s.category].tools[setup.cand[i][k]].text(setup.max_chars) for k in top]
        keys = [text_key(model_name, s.situation, d) for d in docs]
        vals, missing = store.get_many(keys)
        out[i] = (top, vals[:, 0])
        todo += [(i, m, keys[m], (s.situation, docs[m])) for m in missing]
    if todo:
        t0 = time.time()
        model = CrossEncoder(model_name, device="cpu", max_length=max_length)
        if model.model.device.type == "cpu":
            model.model.float()
        for b in range(0, len(todo), 512):
            chunk = todo[b:b + 512]
            pred = np.asarray(model.predict([p for *_, p in chunk], batch_size=32), np.float32).reshape(-1)
            store.put_many([k for _, _, k, _ in chunk], pred)
            for (i, m, _, _), v in zip(chunk, pred):
                out[i][1][m] = v
            store.save()
            print(f"{model_name}: {b + len(chunk)}/{len(todo)} pairs, {time.time() - t0:.0f}s", flush=True)
    return out


def adaptive_k(setup: Setup, ranks: np.ndarray, scores: list[np.ndarray], target: float) -> dict:
    """Coverage and candidate count of fixed and adaptive K, each calibrated on the other folds."""
    n = len(ranks)
    gap = np.array([s.max() - s[g] for s, g in zip(scores, setup.gold)])
    rows = {m: (np.zeros(n, bool), np.zeros(n)) for m in
            ("fixed K", "K per toolset", "score margin", "score margin per toolset")}

    def margin_set(i, d):
        return int((scores[i] >= scores[i].max() - d).sum()) if math.isfinite(d) else int(setup.size[i])

    for f in np.unique(setup.fold):
        test, cal = np.flatnonzero(setup.fold == f), np.flatnonzero(setup.fold != f)
        k = conformal_q(ranks[cal] + 1, target)  # smallest K covering the calibration ranks
        d = conformal_q(gap[cal], target)
        for i in test:
            rows["fixed K"][0][i], rows["fixed K"][1][i] = ranks[i] < k, min(k, setup.size[i])
            rows["score margin"][0][i] = gap[i] <= d
            rows["score margin"][1][i] = margin_set(i, d)
        for ts in np.unique(setup.toolset[test]):
            t_test = test[setup.toolset[test] == ts]
            t_cal = cal[setup.toolset[cal] == ts]
            k_t = conformal_q(ranks[t_cal] + 1, target)
            d_t = conformal_q(gap[t_cal], target)
            for i in t_test:
                rows["K per toolset"][0][i] = ranks[i] < k_t
                rows["K per toolset"][1][i] = min(k_t, setup.size[i])
                rows["score margin per toolset"][0][i] = gap[i] <= d_t
                rows["score margin per toolset"][1][i] = margin_set(i, d_t)
    return {m: {"coverage": float(c.mean()), "mean_k": float(s.mean()), "median_k": float(np.median(s)),
                "p90_k": float(np.percentile(s, 90))} for m, (c, s) in rows.items()}


def summary(ranks: np.ndarray, base: np.ndarray | None, target: float) -> dict:
    out = {f"R@{k}": float(np.mean(ranks < k)) for k in KS}
    out["K for target"] = smallest_k(ranks, target)
    if base is not None:
        out["dR@50"] = paired_bootstrap((ranks < 50).astype(float), (base < 50).astype(float))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data/toolret"))
    p.add_argument("--cache", type=Path, default=Path("cache"))
    p.add_argument("--out", type=Path, default=Path("results/prefilter_k50"))
    p.add_argument("--categories", default="code,customized")
    p.add_argument("--base-encoders", default="Lux1997/Tool-Embed-0.6B,mangopy/ToolRet-trained-bge-base-en-v1.5")
    p.add_argument("--extra-encoders", default="mangopy/ToolRet-trained-bge-large-en-v1.5,"
                                               "mangopy/ToolRet-trained-e5-large-v2")
    p.add_argument("--finetune", default="mangopy/ToolRet-trained-bge-base-en-v1.5",
                   help="comma list of encoders to fine-tune in-domain (cross-fitted); empty to skip")
    p.add_argument("--rerankers", default="")
    p.add_argument("--rerank-n", type=int, default=120)
    p.add_argument("--rerank-depth", type=int, default=200)
    p.add_argument("--rerank-max-length", type=int, default=192)
    p.add_argument("--max-chars", type=int, default=600)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--target", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    setup = Setup(args)
    n = len(setup.steps)
    print(f"{n} single-tool steps, {len(set(setup.toolset))} toolsets", flush=True)
    base_names = args.base_encoders.split(",")
    extra = [e for e in args.extra_encoders.split(",") if e]
    tune = [e for e in args.finetune.split(",") if e]
    sims, q, q_all, tool_emb = {}, {}, {}, {}
    for name in dict.fromkeys(base_names + extra + tune):
        sims[name], q[name], q_all[name], tool_emb[name] = encoder_sims(setup, name, args.cache)
    for name in tune:
        sims[f"ft:{name}"] = finetune_sims(setup, name, q_all[name], tool_emb[name], args.cache, args.seed)
    short = lambda e: ("fine-tuned " if e.startswith("ft:") else "") + e.split("/")[-1]
    r50 = lambda v: np.mean([gold_rank(s, g) < 50 for s, g in zip(v, setup.gold)])

    def ensemble(names):  # RRF over encoders, ties broken by mean cosine (as HybridPrefilter)
        return [sum(rrf(sims[e][i]) for e in names) + 1e-6 * np.mean([sims[e][i] for e in names], 0)
                for i in range(n)]

    methods: dict[str, list[np.ndarray]] = {}
    for e in sims:
        methods[short(e)] = sims[e]
    base_key = "base: " + " + ".join(map(short, base_names))
    methods[base_key] = ensemble(base_names)
    for e in extra + [f"ft:{t}" for t in tune]:
        methods[f"base + {short(e)}"] = ensemble(base_names + [e])
    if len(extra) > 1:
        methods["base + " + " + ".join(map(short, extra))] = ensemble(base_names + extra)
    for t in tune:  # the fine-tuned encoder in place of its original
        swapped = [f"ft:{t}" if e == t else e for e in base_names]
        if swapped != base_names:
            methods["base with " + short(f"ft:{t}")] = ensemble(swapped)

    usage = usage_scores(setup, [q[e] for e in base_names], [q_all[e] for e in base_names])
    has_hist = np.array([np.isfinite(u[g]) for u, g in zip(usage, setup.gold)])
    methods["usage kNN only"] = [np.where(np.isfinite(u), u, -10.0) for u in usage]
    best_dense = max((k for k in methods if k.startswith("base")), key=lambda k: r50(methods[k]))
    gates = {}
    for key in dict.fromkeys([base_key, best_dense]):
        methods[f"{key} + usage kNN (plain RRF)"] = [b + rrf_partial(u) for b, u in zip(methods[key], usage)]
        methods[f"{key} + usage kNN (gated)"], gates[key] = gated_usage(setup, methods[key], usage)

    ranks = {m: np.array([gold_rank(s, g) for s, g in zip(v, setup.gold)]) for m, v in methods.items()}
    base_r = ranks[base_key]
    res = {"n": n, "categories": args.categories.split(","), "target": args.target, "folds": args.folds,
           "base": base_key, "gold_tool_used_before": float(has_hist.mean()), "usage_gates": gates,
           "methods": {m: summary(r, None if m == base_key else base_r, args.target) for m, r in ranks.items()}}
    subsets = sorted({s.subset for s in setup.steps})
    sub = np.array([s.subset for s in setup.steps])
    res["by_subset"] = {m: {x: float(np.mean(ranks[m][sub == x] < 50)) for x in subsets} for m in ranks}
    res["subset_n"] = {x: int((sub == x).sum()) for x in subsets}
    best = max(ranks, key=lambda m: np.mean(ranks[m] < 50))
    res["best"] = best
    res["adaptive"] = {m: adaptive_k(setup, ranks[m], methods[m], args.target)
                       for m in dict.fromkeys([base_key, best])}
    # The margin rule over mean cosine (a calibrated score scale) for the base encoders.
    mean_cos = [np.mean([sims[e][i] for e in base_names], 0) for i in range(n)]
    res["adaptive"]["mean cosine of the base encoders"] = adaptive_k(
        setup, np.array([gold_rank(s, g) for s, g in zip(mean_cos, setup.gold)]), mean_cos, args.target)

    rerankers = [r for r in args.rerankers.split(",") if r]
    if rerankers:
        rows = np.array(sorted(setup.steps.index(s) for s in stratified_sample(setup.steps, args.rerank_n, args.seed)))
        res["rerank"] = {"n": len(rows), "depth": args.rerank_depth, "rows": {}}
        for first in [best]:
            res["rerank"]["rows"][first] = {"first stage": summary(ranks[first][rows], None, args.target)}
        for rr in rerankers:
            for first in [best]:
                ce = rerank_scores(setup, rows, methods[first], rr, args.rerank_depth, args.cache,
                                   args.rerank_max_length)
                only, fused = [], []
                for i in rows:
                    top, v = ce[i]
                    s1 = methods[first][i].copy()
                    s2 = methods[first][i].copy()
                    first_rrf = 1.0 / (RRF_K + 1 + np.arange(len(top)))
                    s1[top] = 2.0 + 1.0 / (1 + desc_rank(v))
                    s2[top] = 2.0 + first_rrf + 1.0 / (RRF_K + 1 + desc_rank(v))
                    only.append(gold_rank(s1, setup.gold[i]))
                    fused.append(gold_rank(s2, setup.gold[i]))
                base_rows = ranks[first][rows]
                res["rerank"]["rows"][first][f"{short(rr)} rerank"] = summary(np.array(only), base_rows, args.target)
                res["rerank"]["rows"][first][f"{short(rr)} RRF with first stage"] = summary(
                    np.array(fused), base_rows, args.target)
                print(f"{rr} over {first}: R@50 {np.mean(np.array(only) < 50):.3f} rerank, "
                      f"{np.mean(np.array(fused) < 50):.3f} fused, first stage {np.mean(base_rows < 50):.3f}",
                      flush=True)
    write_json(args.out / "results.json", res)
    write_report(res, args.out, short)
    print(f"wrote {args.out / 'report.md'}")


def write_report(res: dict, out: Path, short) -> None:
    ci = lambda d: f"{d['mean_diff']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]" if d else "–"
    row = lambda name, r: [name, *(r[f"R@{k}"] for k in KS), r["K for target"], ci(r.get("dR@50"))]
    head = ["method", *(f"R@{k}" for k in KS), f"K for {res['target']:.0%}", "ΔR@50 vs base [95% CI]"]
    methods = sorted(res["methods"].items(), key=lambda kv: -kv[1]["R@50"])
    lines = [
        "# Prefilter at small K",
        "",
        f"{res['n']} single-tool steps from ToolRet {', '.join(res['categories'])}. Each step's own toolset is "
        "ranked (the `toolset` scope); recall@K = the gold tool is in the top K. Base = RRF ensemble of the "
        "encoders over Tool-REX expanded docs (the section 7 prefilter, with ranks taken within the toolset). "
        f"Usage kNN = query-to-query kNN over past requests from the other {res['folds']} cross-fitting folds "
        f"of the same toolset (same-text requests left out); the gold tool had been used before for "
        f"{res['gold_tool_used_before']:.0%} of steps. ΔR@50 = paired bootstrap against the base.",
        "",
        md_table(head, [row(m, r) for m, r in methods]),
        "",
        "## Recall@50 by dataset",
        "",
        md_table(["dataset", "n", *[m for m, _ in methods[:6]]],
                 [[x, res["subset_n"][x], *(res["by_subset"][m][x] for m, _ in methods[:6])]
                  for x in res["subset_n"]]),
        "",
        "## Adaptive K",
        "",
        f"Candidates per request calibrated for {res['target']:.0%} coverage on the other folds "
        "(split conformal, cross-fitted). `fixed K`: one K for all; `K per toolset`: one K per toolset "
        "(Mondrian conformal); `score margin`: keep every tool whose score is within a calibrated margin "
        "of the request's best tool, globally or per toolset. K is capped at the toolset size.",
        "",
        md_table(["ranking", "rule", "coverage", "mean K", "median K", "90th pct K"],
                 [[m, rule, v["coverage"], v["mean_k"], v["median_k"], v["p90_k"]]
                  for m, rules in res["adaptive"].items() for rule, v in rules.items()]),
    ]
    if "rerank" in res:
        rr = res["rerank"]
        lines += ["", "## Cross-encoder reranking", "",
                  f"Stratified sample of {rr['n']} steps; the top {rr['depth']} candidates of the first stage are "
                  "reordered by the cross-encoder (`rerank`) or by RRF of both orders. ΔR@50 = paired bootstrap "
                  "against the first stage on the same steps.", ""]
        for first, rows in rr["rows"].items():
            lines += [f"First stage: {first}", "", md_table(head, [row(m, r) for m, r in rows.items()]), ""]
    (out / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
