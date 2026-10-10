"""Stage-2 score sources: map each step's K candidates to probabilities.

Two modes, as in design Section 7.5:
- "choice":    one distribution over the K candidates (sums to at most 1)  -> LAC / APS
- "predicate": an independent probability per candidate                   -> conformal risk control

Local sources (BM25, dense, cross-encoder) produce raw logits, calibrated on the dev
split with temperature scaling (choice) or Platt scaling (predicate). API sources in
`openai_scorers.py` return probabilities directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, log_softmax

from .cache import ArrayCache, text_key
from .retrieval import Pool, default_device


class Source(Protocol):
    name: str
    mode: str  # "choice" | "predicate"

    def fit(self, dev: Pool) -> None: ...

    def probs(self, pool: Pool) -> np.ndarray: ...


# ---- Calibration ----

def fit_temperature(logits: np.ndarray, gold_mask: np.ndarray) -> float:
    """Temperature T minimising the NLL of the gold mass among the candidates."""
    rows = gold_mask.any(1)
    z, g = logits[rows], gold_mask[rows]

    def nll(log_t: float) -> float:
        lp = log_softmax(z / np.exp(log_t), axis=1)
        return -np.logaddexp.reduce(np.where(g, lp, -np.inf), axis=1).mean()

    return float(np.exp(minimize_scalar(nll, bounds=(-6, 6), method="bounded").x))


def softmax_t(logits: np.ndarray, t: float) -> np.ndarray:
    return np.exp(log_softmax(logits / t, axis=1))


def fit_platt(logits: np.ndarray, gold_mask: np.ndarray) -> tuple[float, float]:
    x, y = logits.ravel(), gold_mask.ravel().astype(float)
    mu, sd = x.mean(), x.std() + 1e-9

    def bce(ab):
        p = np.clip(expit(ab[0] * (x - mu) / sd + ab[1]), 1e-9, 1 - 1e-9)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()

    a, b = minimize(bce, np.array([1.0, 0.0]), method="L-BFGS-B").x
    return float(a / sd), float(b - a * mu / sd)


class CalibratedSource:
    """Wraps a raw-logit scorer and calibrates it on the dev split."""

    def __init__(self, name: str, logit_fn, mode: str):
        self.name, self.logit_fn, self.mode = name, logit_fn, mode
        self.params: tuple[float, ...] = ()

    def fit(self, dev: Pool) -> None:
        z = self.logit_fn(dev)
        self.params = (fit_temperature(z, dev.gold_mask),) if self.mode == "choice" else fit_platt(z, dev.gold_mask)

    def probs(self, pool: Pool) -> np.ndarray:
        z = self.logit_fn(pool)
        if self.mode == "choice":
            return softmax_t(z, self.params[0])
        a, b = self.params
        return expit(a * z + b)


def feature_logits(feature: str):
    """Use a Stage-1 retriever signal (bm25, dense cosine, rrf) as the Stage-2 score."""
    return lambda pool: getattr(pool, feature).astype(np.float64)


class CrossEncoderLogits:
    """Cross-encoder reranker logits for (situation, tool) pairs, cached per pair."""

    def __init__(self, model_name: str, cache_dir: Path, batch_size: int = 128, max_chars: int = 400):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=default_device())
        self.batch_size, self.max_chars = batch_size, max_chars
        self.cache = ArrayCache(cache_dir / f"ce_{model_name.replace('/', '__')}.npz", 1)

    def __call__(self, pool: Pool) -> np.ndarray:
        pairs = [(s.situation, t.text(self.max_chars)) for s, row in zip(pool.steps, pool.tools) for t in row]
        keys = [text_key(a, b) for a, b in pairs]
        vals, missing = self.cache.get_many(keys)
        if missing:
            new = self.model.predict([pairs[i] for i in missing], batch_size=self.batch_size,
                                     show_progress_bar=len(missing) > 5000, convert_to_numpy=True)
            vals[missing, 0] = new
            self.cache.put_many([keys[i] for i in missing], new)
            self.cache.save()
        return vals[:, 0].reshape(len(pool), pool.k).astype(np.float64)
