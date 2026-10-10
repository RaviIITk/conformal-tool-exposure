"""Metrics and statistics (design Sections 11-12)."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def set_metrics(sets: np.ndarray, gold_mask: np.ndarray, n_gold: np.ndarray) -> dict:
    covered = (sets & gold_mask).sum(1)
    size = sets.sum(1)
    return {
        "coverage": float((covered == n_gold).mean()),
        "fnr": float((1 - covered / n_gold).mean()),
        "avg_size": float(size.mean()),
        "median_size": float(np.median(size)),
        "singleton_rate": float((size == 1).mean()),
        "empty_rate": float((size == 0).mean()),
    }


def recall_at_k(gold_mask: np.ndarray, n_gold: np.ndarray) -> float:
    """Share of steps whose gold tools all survive the prefilter (hard ceiling on coverage)."""
    return float((gold_mask.sum(1) == n_gold).mean())


def top1_accuracy(probs: np.ndarray, gold_mask: np.ndarray) -> float:
    return float(gold_mask[np.arange(len(probs)), probs.argmax(1)].mean())


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def calibration_error(probs: np.ndarray, gold_mask: np.ndarray, mode: str) -> float:
    """Choice: ECE of top-1 confidence. Predicate: ECE over every (step, candidate) pair."""
    if mode == "choice":
        top = probs.argmax(1)
        return ece(probs[np.arange(len(probs)), top], gold_mask[np.arange(len(probs)), top].astype(float))
    return ece(probs.ravel(), gold_mask.ravel().astype(float))


def reliability_curve(conf: np.ndarray, correct: np.ndarray, bins: int = 10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    xs, ys, ns = [], [], []
    for b in range(bins):
        m = idx == b
        if m.any():
            xs.append(conf[m].mean()), ys.append(correct[m].mean()), ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def split_coverage_theory(n_cal: int, alpha: float) -> dict:
    """Distribution of coverage given a calibration set under exchangeability: Beta(n+1-l, l)."""
    l = math.floor((n_cal + 1) * alpha)
    if l < 1:
        return {"mean": 1.0, "p05": 1.0}
    b = stats.beta(n_cal + 1 - l, l)
    return {"mean": float(b.mean()), "p05": float(b.ppf(0.05))}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """CI and two-sided p-value for mean(a - b) over paired observations."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"mean_diff": float(d.mean()), "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)), "p": float(min(1.0, max(p, 1 / n_boot)))}


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values."""
    order = np.argsort(pvals)
    m, out, running = len(pvals), [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        out[i] = min(1.0, running)
    return out
