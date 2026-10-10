"""Conformal prediction over Stage-2 tool probabilities (design Section 8).

All functions take `probs` [n, K] for the K prefilter candidates, a boolean `gold_mask`
[n, K] and `n_gold` [n], the total number of gold tools per step. A gold tool missed
by the prefilter gives that step a score of +inf, so the guarantee covers both stages
end to end (Section 8.4).
"""

from __future__ import annotations

import math

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The ceil((n+1)(1-alpha))/n empirical quantile; +inf when n is too small."""
    n = len(scores)
    level = math.ceil((n + 1) * (1 - alpha))
    if level > n:
        return math.inf
    return float(np.sort(scores)[level - 1])


def _prefilter_miss(gold_mask: np.ndarray, n_gold: np.ndarray) -> np.ndarray:
    return gold_mask.sum(1) < n_gold


# ---- LAC (Mode A, single-label score extended to "all gold tools covered") ----

def lac_scores(probs: np.ndarray, gold_mask: np.ndarray, n_gold: np.ndarray) -> np.ndarray:
    """s = max over gold tools of 1 - p(tool); equals plain LAC when a step has one gold tool."""
    s = np.where(gold_mask, 1.0 - probs, -np.inf).max(1)
    s[_prefilter_miss(gold_mask, n_gold)] = np.inf
    return s


def lac_sets(probs: np.ndarray, qhat: float) -> np.ndarray:
    return (1.0 - probs) <= qhat


# ---- APS (randomized adaptive prediction sets, Romano et al. 2020) ----

def _aps_cum_before(probs: np.ndarray) -> np.ndarray:
    """For each candidate, the probability mass ranked strictly above it."""
    order = np.argsort(-probs, axis=1, kind="stable")
    sorted_p = np.take_along_axis(probs, order, 1)
    before_sorted = np.cumsum(sorted_p, 1) - sorted_p
    before = np.empty_like(probs)
    np.put_along_axis(before, order, before_sorted, 1)
    return before


def aps_scores(probs, gold_mask, n_gold, u: np.ndarray) -> np.ndarray:
    e = _aps_cum_before(probs) + u[:, None] * probs
    s = np.where(gold_mask, e, -np.inf).max(1)
    s[_prefilter_miss(gold_mask, n_gold)] = np.inf
    return s


def aps_sets(probs: np.ndarray, qhat: float, u: np.ndarray) -> np.ndarray:
    return (_aps_cum_before(probs) + u[:, None] * probs) <= qhat


# ---- Conformal risk control (Mode B, multi-tool steps; Angelopoulos et al. 2024) ----

def fnr_loss(sets: np.ndarray, gold_mask: np.ndarray, n_gold: np.ndarray) -> np.ndarray:
    """Fraction of a step's gold tools missing from its set (prefilter misses count as missing)."""
    return 1.0 - (sets & gold_mask).sum(1) / n_gold


def crc_threshold(probs, gold_mask, n_gold, alpha: float) -> float:
    """Largest probability threshold tau with (n/(n+1)) * mean FNR + 1/(n+1) <= alpha.

    The set is {t : p_t >= tau}. The empirical risk only changes at gold-tool
    probabilities, so it is evaluated exactly at those values (sorted, cumulative).
    Returns 0.0 (keep every candidate) when no tau meets the bound, which happens when
    prefilter misses alone push the risk above alpha.
    """
    n = len(probs)
    rows, _ = np.nonzero(gold_mask)
    p = probs[gold_mask]
    order = np.argsort(-p, kind="stable")
    p_sorted = p[order]
    covered = np.cumsum(1.0 / n_gold[rows][order])  # sum over steps of covered gold fraction
    # With ties, a threshold at p_sorted[i] includes every gold tool with p >= p_sorted[i].
    last_of_tie = np.searchsorted(-p_sorted, -p_sorted, side="right") - 1
    risk = 1.0 - covered[last_of_tie] / n
    ok = np.flatnonzero(n / (n + 1) * risk + 1 / (n + 1) <= alpha)
    return float(p_sorted[ok[0]]) if len(ok) else 0.0


def crc_sets(probs: np.ndarray, tau: float) -> np.ndarray:
    return probs >= tau


# ---- Practical adjustments (Section 8.5) ----

def with_top1_fallback(sets: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Abstention handling: an empty set falls back to the top-1 candidate."""
    out = sets.copy()
    empty = ~out.any(1)
    out[empty, probs[empty].argmax(1)] = True
    return out


def cap_sets(sets: np.ndarray, probs: np.ndarray, k_max: int) -> np.ndarray:
    """Keep at most k_max highest-probability members. This breaks the guarantee."""
    masked = np.where(sets, probs, -np.inf)
    keep = np.argsort(-masked, axis=1, kind="stable")[:, :k_max]
    out = np.zeros_like(sets)
    np.put_along_axis(out, keep, True, 1)
    return out & sets


# ---- Online conformal for drift (adaptive conformal inference, Gibbs & Candes 2021) ----

class AdaptiveConformal:
    """Tracks the miscoverage level online from feedback on whether the set covered the step.

    After each step: alpha_t <- alpha_t + gamma * (alpha - err_t). The threshold is the
    1 - alpha_t quantile of a sliding window of recent nonconformity scores, which starts
    as the calibration scores and absorbs each newly observed score.
    """

    def __init__(self, cal_scores: np.ndarray, alpha: float, gamma: float = 0.01, window: int | None = None):
        self.alpha, self.gamma = alpha, gamma
        self.alpha_t = alpha
        self.window = window or len(cal_scores)
        self.scores = list(np.asarray(cal_scores)[-self.window:])

    def qhat(self) -> float:
        if self.alpha_t <= 0:
            return math.inf
        if self.alpha_t >= 1:
            return -math.inf
        return conformal_quantile(np.array(self.scores), self.alpha_t)

    def update(self, score: float) -> bool:
        """Record one step's true score; returns whether the set at the current qhat missed."""
        err = score > self.qhat()
        self.alpha_t += self.gamma * (self.alpha - float(err))
        self.scores.append(score)
        if len(self.scores) > self.window:
            self.scores.pop(0)
        return err
