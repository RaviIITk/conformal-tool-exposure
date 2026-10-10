"""Markdown tables and figures for the experiment reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PALETTE = ["#2a6fdb", "#e0782c", "#2e9d5b", "#c43c5c", "#7a52c7", "#8a8a8a", "#1aa3a3", "#b8860b"]


def fmt(v, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if isinstance(v, float):
        return "∞" if np.isinf(v) else f"{v:.{digits}f}"
    return str(v)


def md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def line_plot(path: Path, series: dict[str, tuple], title: str, xlabel: str, ylabel: str,
              diagonal: bool = False, hline: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for (name, (x, y)), c in zip(series.items(), PALETTE):
        ax.plot(x, y, marker="o", ms=4, color=c, label=name)
    if diagonal:
        lo = min(min(x) for x, _ in series.values())
        ax.plot([lo, 1], [lo, 1], ls="--", color="#999", lw=1, label="target")
    if hline is not None:
        ax.axhline(hline, ls="--", color="#999", lw=1, label="target")
    _style(ax, title, xlabel, ylabel)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def reliability_plot(path: Path, curves: dict[str, tuple]) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="#999", lw=1)
    for (name, (x, y, _)), c in zip(curves.items(), PALETTE):
        ax.plot(x, y, marker="o", ms=4, color=c, label=name)
    _style(ax, "Reliability of top-1 confidence", "confidence", "accuracy")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def coverage_hist(path: Path, covs: list[float], theory_p05: float, target: float, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(covs, bins=20, color=PALETTE[0], alpha=0.8)
    ax.axvline(target, color="#333", ls="--", lw=1, label=f"1-α = {target:.2f}")
    ax.axvline(theory_p05, color=PALETTE[1], ls=":", lw=1.5, label=f"Beta theory 5th pct = {theory_p05:.3f}")
    _style(ax, title, "empirical coverage on test half", "splits")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
