"""Shared setup for the experiment scripts: data -> prefilter -> pool -> dev/eval split -> sources."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import Catalog, Step, expand_tool_docs, load_toolret
from .experiment import choose_k, dev_split, stratified_sample
from .retrieval import Encoder, HybridPrefilter, Pool, build_pool
from .scorers import CalibratedSource, CrossEncoderLogits, feature_logits

LOCAL_SOURCES = ("bm25", "dense", "hybrid", "ce", "ce-pred")
API_SOURCES = ("mercury-choice", "mercury-predicate", "decisions-choice", "decisions-predicate", "logprob")


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", type=Path, default=Path("data/toolret"))
    p.add_argument("--cache", type=Path, default=Path("cache"))
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--encoder", default="Qwen/Qwen3-Embedding-0.6B",
                   help="sentence-transformers model; `A+B` fuses several with RRF; `none` = BM25 only")
    p.add_argument("--prefilter-scope", default="pooled",
                   help="pooled (whole catalog), routed<m> (m best-matching toolsets first) or toolset "
                        "(the step's own dataset toolset first: the tools its agent is connected to)")
    p.add_argument("--no-bm25", action="store_true", help="fuse the encoders only, without BM25")
    p.add_argument("--ce-model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    p.add_argument("--max-steps", type=int, default=None, help="stratified sample of ToolRet steps")
    p.add_argument("--single-tool-only", action="store_true", help="keep only steps with one gold tool")
    p.add_argument("--tool-docs", choices=("toolret", "expanded"), default="toolret",
                   help="tool documents: as in ToolRet, or with Tool-REX profiles appended (document expansion)")
    p.add_argument("--k", type=int, default=None, help="prefilter K (default: chosen on dev)")
    p.add_argument("--k-grid", default="20,50,100")
    p.add_argument("--dev-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sources", default=",".join(LOCAL_SOURCES),
                   help=f"comma list from {LOCAL_SOURCES + API_SOURCES}")


@dataclass
class Setup:
    catalogs: dict[str, Catalog]
    prefilters: dict[str, HybridPrefilter]
    encoder: list[Encoder] | None
    dev: Pool
    eval: Pool
    k: int
    recalls: dict[int, float]


def build_setup(args) -> Setup:
    catalogs, steps = load_toolret(args.data)
    if getattr(args, "tool_docs", "toolret") == "expanded":
        catalogs = expand_tool_docs(catalogs, args.data)
    if getattr(args, "single_tool_only", False):
        steps = [s for s in steps if len(s.gold) == 1]
    steps = stratified_sample(steps, args.max_steps, args.seed)
    encoder = [Encoder(n, args.cache) for n in args.encoder.split("+")] if args.encoder != "none" else None
    prefilters = {c: HybridPrefilter(cat, encoder, scope=getattr(args, "prefilter_scope", "pooled"),
                                     use_bm25=not getattr(args, "no_bm25", False)) for c, cat in catalogs.items()}
    k_grid = [int(x) for x in args.k_grid.split(",")]
    pool = build_pool(steps, prefilters, max(k_grid + [args.k or 0]))
    dev_rows, eval_rows = dev_split(len(pool), args.dev_frac, args.seed)
    dev, ev = pool.take(dev_rows), pool.take(eval_rows)
    k_auto, recalls = choose_k(dev, k_grid, alpha=0.1, margin=0.02)
    k = args.k or k_auto
    return Setup(catalogs, prefilters, encoder, dev.truncate(k), ev.truncate(k), k, recalls)


def make_source(name: str, args):
    """Build one Stage-2 score source by name (API sources import lazily)."""
    if name in ("bm25", "dense"):
        return CalibratedSource(name, feature_logits(name), "choice")
    if name == "hybrid":
        return CalibratedSource(name, feature_logits("rrf"), "choice")
    if name in ("ce", "ce-pred"):
        if not hasattr(args, "_ce"):
            args._ce = CrossEncoderLogits(args.ce_model, args.cache)
        return CalibratedSource(name, args._ce, "predicate" if name == "ce-pred" else "choice")
    from .openai_scorers import DecisionsSource, LogprobSource, SystemOneSource

    if name.startswith("mercury-"):
        return SystemOneSource(name.split("-", 1)[1], args.cache)
    if name.startswith("decisions-"):
        return DecisionsSource(name.split("-", 1)[1], args.cache)
    if name == "logprob":
        return LogprobSource(args.cache)
    raise ValueError(f"unknown source {name}")


def subset_by(pool: Pool, mask: np.ndarray) -> Pool:
    return pool.take(np.flatnonzero(mask))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o))

    path.write_text(json.dumps(obj, indent=2, default=default))
