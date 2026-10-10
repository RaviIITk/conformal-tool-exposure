"""Stage 1 prefilter: BM25 + dense retrieval fused with Reciprocal Rank Fusion (as in SCOUT)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import bm25s
import numpy as np

from .cache import ArrayCache, text_key
from .data import Catalog, Step, Tool


TOOL_QUERY_INSTRUCTION = ("Instruct: Given a description of what an AI agent needs to do next, "
                          "retrieve the tool that can perform it\nQuery:")


def default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Encoder:
    """Sentence-transformers bi-encoder with an on-disk embedding cache."""

    def __init__(self, model_name: str, cache_dir: Path, device: str | None = None, batch_size: int = 64,
                 max_seq_length: int = 256, trust_remote_code: bool = False):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name, device=device or default_device(),
                                         trust_remote_code=trust_remote_code)
        if self.model.device.type == "cpu":
            self.model.float()  # bf16/fp16 checkpoints run many times slower on CPUs without native bf16
        self.model.max_seq_length = max_seq_length
        self.batch_size = batch_size
        # Asymmetric embedders take a query prompt and sometimes a document prompt (e.g. LFM2.5:
        # "query: " / "document: "). Instruction-tuned ones (e.g. Qwen3-Embedding) take a task
        # instruction on queries, which is replaced by a tool-retrieval one.
        prompts = self.model.prompts or {}
        self.query_prompt = self.doc_prompt = None
        if prompts.get("query"):
            if prompts["query"].startswith("Instruct:"):
                prompts["query"] = TOOL_QUERY_INSTRUCTION
            self.query_prompt = "query"
        if prompts.get("document"):
            self.doc_prompt = "document"
        dim = self.model.get_embedding_dimension()
        self.cache = ArrayCache(cache_dir / f"emb_{model_name.replace('/', '__')}.npz", dim)

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        prompt = self.query_prompt if is_query else self.doc_prompt
        keys = [text_key(t, self.model.prompts[prompt]) if prompt else text_key(t) for t in texts]
        out, missing = self.cache.get_many(keys)
        if missing:
            vecs = self.model.encode([texts[i] for i in missing], batch_size=self.batch_size,
                                     prompt_name=prompt, normalize_embeddings=True, convert_to_numpy=True,
                                     show_progress_bar=len(missing) > 2000)
            out[missing] = vecs
            self.cache.put_many([keys[i] for i in missing], vecs)
            if len(missing) > 100:
                self.cache.save()
        return out


@dataclass
class Pool:
    """Steps with their Stage-1 candidates, ready for Stage-2 scoring.

    Candidates are sorted best-first by RRF. `n_gold` counts every gold tool, including
    the ones the prefilter missed, so prefilter misses count as non-coverage downstream.
    """

    steps: list[Step]
    tools: list[list[Tool]]
    bm25: np.ndarray
    dense: np.ndarray
    rrf: np.ndarray
    gold_mask: np.ndarray
    n_gold: np.ndarray

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def k(self) -> int:
        return self.gold_mask.shape[1]

    @property
    def hit_all(self) -> np.ndarray:
        """True where every gold tool survived the prefilter."""
        return self.gold_mask.sum(1) == self.n_gold

    def take(self, rows) -> Pool:
        rows = np.asarray(rows)
        return Pool([self.steps[i] for i in rows], [self.tools[i] for i in rows], self.bm25[rows],
                    self.dense[rows], self.rrf[rows], self.gold_mask[rows], self.n_gold[rows])

    def truncate(self, k: int) -> Pool:
        return replace(self, tools=[t[:k] for t in self.tools], bm25=self.bm25[:, :k],
                       dense=self.dense[:, :k], rrf=self.rrf[:, :k], gold_mask=self.gold_mask[:, :k])

    @staticmethod
    def concat(pools: list[Pool]) -> Pool:
        pools = [p for p in pools if len(p)]
        return Pool(sum((p.steps for p in pools), []), sum((p.tools for p in pools), []),
                    *(np.concatenate([getattr(p, f) for p in pools]) for f in
                      ("bm25", "dense", "rrf", "gold_mask", "n_gold")))


class HybridPrefilter:
    def __init__(self, catalog: Catalog, encoder: Encoder | None, rrf_k: int = 60, depth: int = 200,
                 max_chars: int = 400):
        self.catalog, self.encoder, self.rrf_k, self.depth = catalog, encoder, rrf_k, depth
        texts = [t.text(max_chars) for t in catalog.tools]
        self.bm25 = bm25s.BM25()
        self.bm25.index(bm25s.tokenize(texts, stopwords="en", show_progress=False), show_progress=False)
        self.emb = encoder.encode(texts) if encoder else None

    def _bm25_scores(self, situations: list[str]) -> np.ndarray:
        tokens = bm25s.tokenize(situations, stopwords="en", return_ids=False, show_progress=False)
        out = np.zeros((len(situations), len(self.catalog)), dtype=np.float32)
        for i, toks in enumerate(tokens):
            toks = [t for t in toks if t in self.bm25.vocab_dict]
            if toks:
                out[i] = self.bm25.get_scores(toks)
        return out

    def _ranks_to_rrf(self, scores: np.ndarray) -> np.ndarray:
        depth = min(self.depth, scores.shape[1])
        top = np.argpartition(-scores, depth - 1, axis=1)[:, :depth]
        order = np.argsort(-np.take_along_axis(scores, top, 1), axis=1)
        top = np.take_along_axis(top, order, 1)
        rrf = np.zeros_like(scores)
        rows = np.arange(len(scores))[:, None]
        rrf[rows, top] = 1.0 / (self.rrf_k + np.arange(1, depth + 1))
        # Documents with zero lexical overlap carry no BM25 evidence.
        rrf[scores <= 0] = 0.0
        return rrf

    def search(self, steps: list[Step], k: int, batch: int = 256) -> Pool:
        tools = self.catalog.tools
        parts = {f: [] for f in ("idx", "bm25", "dense", "rrf")}
        for s in range(0, len(steps), batch):
            sits = [st.situation for st in steps[s:s + batch]]
            bm = self._bm25_scores(sits)
            rrf = self._ranks_to_rrf(bm)
            if self.emb is not None:
                dense = self.encoder.encode(sits, is_query=True) @ self.emb.T
                rrf += self._ranks_to_rrf(dense + 1.0)  # shift so all cosines are positive
            else:
                dense = np.zeros_like(bm)
            fused = rrf + 1e-6 * dense
            top = np.argpartition(-fused, k - 1, axis=1)[:, :k]
            top = np.take_along_axis(top, np.argsort(-np.take_along_axis(fused, top, 1), axis=1), 1)
            parts["idx"].append(top)
            for name, mat in (("bm25", bm), ("dense", dense), ("rrf", rrf)):
                parts[name].append(np.take_along_axis(mat, top, 1))
        idx = np.concatenate(parts["idx"])
        gold_mask = np.array([[tools[j].id in set(st.gold) for j in row] for st, row in zip(steps, idx)])
        return Pool(steps, [[tools[j] for j in row] for row in idx],
                    *(np.concatenate(parts[f]) for f in ("bm25", "dense", "rrf")),
                    gold_mask, np.array([len(st.gold) for st in steps]))


def build_pool(steps: list[Step], prefilters: dict[str, HybridPrefilter], k: int) -> Pool:
    """Run each step through its category's prefilter, keeping the original step order."""
    order, pools = [], []
    for cat, pf in prefilters.items():
        rows = [i for i, s in enumerate(steps) if s.category == cat]
        if rows:
            order += rows
            pools.append(pf.search([steps[i] for i in rows], k))
    pool = Pool.concat(pools)
    return pool.take(np.argsort(order))
