import math

import numpy as np
import pytest

from cte import conformal as cp
from cte.metrics import set_metrics


def synthetic(n=4000, k=20, seed=0, n_gold=1):
    """Well-specified data: gold tools are drawn from the model's own probabilities."""
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.full(k, 0.3), size=n)
    gold = np.zeros((n, k), bool)
    for i in range(n):
        gold[i, rng.choice(k, n_gold, replace=False, p=probs[i])] = True
    return probs, gold, np.full(n, n_gold)


def split_coverage(method, alpha, n_splits=50, **kw):
    probs, gold, ng = synthetic(**kw)
    rng = np.random.default_rng(1)
    covs, sizes = [], []
    for _ in range(n_splits):
        perm = rng.permutation(len(probs))
        cal, test = perm[: len(perm) // 2], perm[len(perm) // 2:]
        if method == "lac":
            q = cp.conformal_quantile(cp.lac_scores(probs[cal], gold[cal], ng[cal]), alpha)
            sets = cp.lac_sets(probs[test], q)
        else:
            q = cp.conformal_quantile(cp.aps_scores(probs[cal], gold[cal], ng[cal], rng.random(len(cal))), alpha)
            sets = cp.aps_sets(probs[test], q, rng.random(len(test)))
        m = set_metrics(sets, gold[test], ng[test])
        covs.append(m["coverage"])
        sizes.append(m["avg_size"])
    return np.mean(covs), np.mean(sizes)


def test_quantile_definition():
    s = np.arange(1, 10, dtype=float)  # n = 9
    assert cp.conformal_quantile(s, 0.1) == 9.0  # ceil(10 * 0.9) = 9 -> 9th smallest
    assert cp.conformal_quantile(s, 0.3) == 7.0
    assert math.isinf(cp.conformal_quantile(s, 0.05))  # ceil(10 * 0.95) = 10 > n


@pytest.mark.parametrize("method", ["lac", "aps"])
@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_marginal_coverage_guarantee(method, alpha):
    cov, size = split_coverage(method, alpha)
    assert cov >= 1 - alpha - 0.01
    assert cov <= 1 - alpha + 0.02  # split CP is also nearly tight
    assert size < 20


def test_lac_gives_smaller_sets_than_aps():
    _, lac = split_coverage("lac", 0.1)
    _, aps = split_coverage("aps", 0.1)
    assert lac < aps


def test_multi_gold_lac_covers_all_gold():
    cov, _ = split_coverage("lac", 0.1, n_gold=3)
    assert cov >= 0.89


def test_prefilter_misses_cap_coverage():
    probs, gold, ng = synthetic(n=1000)
    miss = np.random.default_rng(2).random(1000) < 0.3
    gold[miss] = False  # gold tool not among the K candidates
    scores = cp.lac_scores(probs, gold, ng)
    assert np.isinf(scores[miss]).all()
    q = cp.conformal_quantile(scores, 0.1)
    assert math.isinf(q)  # 30% misses > alpha: keep every candidate
    sets = cp.lac_sets(probs, q)
    assert sets.all()
    assert set_metrics(sets, gold, ng)["coverage"] == pytest.approx(0.7, abs=0.05)


def test_crc_controls_false_negative_rate():
    probs, gold, ng = synthetic(n=4000, n_gold=3)
    rng = np.random.default_rng(3)
    fnrs = []
    for _ in range(30):
        perm = rng.permutation(len(probs))
        cal, test = perm[:2000], perm[2000:]
        tau = cp.crc_threshold(probs[cal], gold[cal], ng[cal], 0.1)
        fnrs.append(cp.fnr_loss(cp.crc_sets(probs[test], tau), gold[test], ng[test]).mean())
    assert np.mean(fnrs) <= 0.1 + 0.005
    assert np.mean(fnrs) >= 0.07  # and not wastefully conservative


def test_crc_threshold_matches_brute_force():
    probs, gold, ng = synthetic(n=300, n_gold=2, seed=5)
    gold[:20] = False  # some prefilter misses
    n = len(probs)
    for alpha in (0.1, 0.2, 0.4):
        best = 0.0
        for tau in np.unique(np.append(probs, 0.0)):
            risk = cp.fnr_loss(probs >= tau, gold, ng).mean()
            if n / (n + 1) * risk + 1 / (n + 1) <= alpha:
                best = max(best, tau)
        assert cp.crc_threshold(probs, gold, ng, alpha) == pytest.approx(best)


def test_aci_restores_coverage_after_shift():
    rng = np.random.default_rng(4)
    cal = rng.random(1000)
    test = rng.random(4000) + 0.25  # scores drift upwards after calibration
    q = cp.conformal_quantile(cal, 0.1)
    static_cov = (test <= q).mean()
    aci = cp.AdaptiveConformal(cal, 0.1, gamma=0.01)
    aci_cov = 1 - np.mean([aci.update(s) for s in test])
    assert static_cov < 0.8
    assert aci_cov == pytest.approx(0.9, abs=0.02)


def test_fallback_and_cap():
    probs = np.array([[0.6, 0.3, 0.1], [0.4, 0.35, 0.25]])
    sets = np.array([[False, False, False], [True, True, True]])
    fb = cp.with_top1_fallback(sets, probs)
    assert fb[0].tolist() == [True, False, False]
    capped = cp.cap_sets(fb, probs, 2)
    assert capped[1].tolist() == [True, True, False]
