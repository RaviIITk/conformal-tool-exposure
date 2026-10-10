"""Pre-compute and cache embeddings for every ToolRet tool and query (resumable, saves every chunk).

  python scripts/warm_embeddings.py --encoder Qwen/Qwen3-Embedding-0.6B
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from cte.data import load_toolret
from cte.retrieval import Encoder


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--data", type=Path, default=Path("data/toolret"))
    p.add_argument("--cache", type=Path, default=Path("cache"))
    p.add_argument("--chunk", type=int, default=2000)
    args = p.parse_args()
    catalogs, steps = load_toolret(args.data)
    enc = Encoder(args.encoder, args.cache, batch_size=32)
    jobs = [([t.text() for c in catalogs.values() for t in c.tools], False), ([s.situation for s in steps], True)]
    for texts, is_query in jobs:
        t0 = time.time()
        for i in range(0, len(texts), args.chunk):
            enc.encode(texts[i:i + args.chunk], is_query=is_query)
            enc.cache.save()
            done = min(i + args.chunk, len(texts))
            print(f"{'queries' if is_query else 'tools'} {done}/{len(texts)} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
