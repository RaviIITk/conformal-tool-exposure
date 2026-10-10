"""Shared experiment plumbing: splits, K selection, repeated split-conformal evaluation."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import conformal as cp
from .data import Step
from .metrics import set_metrics, split_coverage_theory
from .retrieval import Pool


def stratified_sample(steps: list[Step], n: int | None, seed: int = 0) -> list[Step]:
    """Sample n steps keeping each ToolRet subset's share; None keeps everything."""
    if n is None or n >= len(steps):
        return steps
    rng = np.random.default_rng(seed)
    by_subset = defaultdict(list)
    for s in steps:
        by_subset[s.subset].append(s)
    out = []
    for group in by_subset.values():
        take = max(1, round(n * len(group) / len(steps)))
        out += [group[i] for i in rng.choice(len(group), min(take, len(group)), replace=False)]
    if len(out) > n:  # rounding up tiny subsets can overshoot; trim back to exactly n
        out = [out[i] for i in sorted(rng.choice(len(out), n, replace=False))]
    return out


def dev_split(n: int, dev_frac: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Fixed dev rows (choose K, fit calibration) and the rest for repeated cal/test splits.

    Every ToolRet step is an independent task, so splitting by step is splitting by task.
    """
    perm = np.random.default_rng(seed).permutation(n)
    n_dev = int(round(dev_frac * n))
    return np.sort(perm[:n_dev]), np.sort(perm[n_dev:])


def choose_k(dev: Pool, grid: list[int], alpha: float, margin: float) -> tuple[int, dict]:
    """Smallest K whose dev recall@K reaches 1 - alpha + margin, else the largest K."""
    recalls = {k: float((dev.gold_mask[:, :k].sum(1) == dev.n_gold).mean()) for k in grid}
    for k in sorted(grid):
        if recalls[k] >= 1 - alpha + margin:
            return k, recalls
    return max(grid), recalls


def _sets(method: str, probs, gm, ng, cal, test, alpha, rng):
    if method == "lac":
        q = cp.conformal_quantile(cp.lac_scores(probs[cal], gm[cal], ng[cal]), alpha)
        return cp.lac_sets(probs[test], q), q
    if method == "aps":
        q = cp.conformal_quantile(cp.aps_scores(probs[cal], gm[cal], ng[cal], rng.random(len(cal))), alpha)
        return cp.aps_sets(probs[test], q, rng.random(len(test))), q
    if method == "crc":
        tau = cp.crc_threshold(probs[cal], gm[cal], ng[cal], alpha)
        return cp.crc_sets(probs[test], tau), tau
    raise ValueError(method)


def repeated_cp(probs: np.ndarray, pool: Pool, method: str, alpha: float, n_splits: int = 100,
                cal_frac: float = 0.5, k_max: int = 5, seed: int = 0, groups: np.ndarray | None = None) -> dict:
    """Evaluate one CP method over many random calibration/test splits.

    Returns the mean and 5th percentile of each metric across splits, the theoretical
    coverage band, each step's average set size (for paired comparisons) and, when
    `groups` is given, coverage per group (conditional coverage).
    """
    rng = np.random.default_rng(seed)
    gm, ng, n = pool.gold_mask, pool.n_gold, len(pool)
    n_cal = int(cal_frac * n)
    per_split = defaultdict(list)
    size_sum, size_cnt = np.zeros(n), np.zeros(n)
    for _ in range(n_splits):
        perm = rng.permutation(n)
        cal, test = perm[:n_cal], perm[n_cal:]
        sets, thr = _sets(method, probs, gm, ng, cal, test, alpha, rng)
        m = set_metrics(sets, gm[test], ng[test])
        if groups is not None:
            covered = (sets & gm[test]).sum(1) == ng[test]
            for g in np.unique(groups):
                in_g = groups[test] == g
                if in_g.any():
                    m[f"coverage[{g}]"] = float(covered[in_g].mean())
        m["coverage_fallback"] = set_metrics(cp.with_top1_fallback(sets, probs[test]), gm[test], ng[test])["coverage"]
        capped = cp.cap_sets(sets, probs[test], k_max)
        m[f"coverage_capped{k_max}"] = set_metrics(capped, gm[test], ng[test])["coverage"]
        m["threshold_finite"] = float(np.isfinite(thr))
        for key, v in m.items():
            per_split[key].append(v)
        size_sum[test] += sets.sum(1)
        size_cnt[test] += 1
    out = {k: float(np.mean(v)) for k, v in per_split.items()}
    out["coverage_p05"] = float(np.percentile(per_split["coverage"], 5))
    out["coverage_std"] = float(np.std(per_split["coverage"]))
    out["theory"] = split_coverage_theory(n_cal, alpha)
    out["n"] = n
    out["step_avg_size"] = np.divide(size_sum, size_cnt, out=np.full(n, np.nan), where=size_cnt > 0)
    out["coverage_by_split"] = per_split["coverage"]
    return out
