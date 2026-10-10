"""Phase 2 proxy (RQ2): does adaptive exposure beat fixed top-k on success per token?

The agent LLM (OpenAI Chat Completions) sees only the exposed tools as function schemas
and must call the one that performs the step. Success means it calls a gold tool. This
is a single-step proxy for the full multi-step loop on live MCP benchmarks.
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from .cache import JsonlCache, text_key
from .data import Tool
from .openai_scorers import make_client

SYSTEM = "You are an AI agent. Call exactly one tool that performs the user's next action. Do not reply in text."
_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")


def function_schema(tool: Tool, i: int) -> dict:
    """OpenAI function schema; names are sanitised and suffixed with the slot index."""
    name = (_NAME_RE.sub("_", tool.name)[:56] or "tool") + f"_{i}"
    props = {(_NAME_RE.sub("_", p)[:64] or f"arg{j}"): {"type": "string"} for j, p in enumerate(tool.params[:20])}
    return {"type": "function", "function": {"name": name, "description": (tool.description or tool.name)[:1024],
                                             "parameters": {"type": "object", "properties": props}}}


class AgentProxy:
    def __init__(self, cache_dir: Path, model: str | None = None, client=None, workers: int = 8):
        self.model = model or os.environ.get("CTE_AGENT_MODEL", "gpt-5-mini")
        self.cache = JsonlCache(cache_dir / f"openai_{self.model}_agent.jsonl")
        self._client, self.workers = client, workers

    @property
    def client(self):
        if self._client is None:
            self._client = make_client()
        return self._client

    def _request(self, situation: str, tools: list[Tool]) -> dict:
        req = {"model": self.model,
               "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": situation}]}
        if tools:
            req["tools"] = [function_schema(t, i) for i, t in enumerate(tools)]
        return req

    def _call(self, req: dict, n_tools: int) -> dict:
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(**req).model_dump()
        latency = time.perf_counter() - t0
        calls = resp["choices"][0]["message"].get("tool_calls") or []
        chosen = None
        if calls:
            m = re.search(r"_(\d+)$", calls[0]["function"]["name"])
            chosen = int(m.group(1)) if m and int(m.group(1)) < n_tools else None
        usage = resp.get("usage") or {}
        return {"chosen": chosen, "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0), "latency": latency}

    def run(self, situations: list[str], tool_lists: list[list[Tool]], desc: str = "agent") -> list[dict]:
        reqs = [self._request(s, t) for s, t in zip(situations, tool_lists)]
        keys = [text_key(json.dumps(r, sort_keys=True)) for r in reqs]
        todo = [(k, r, len(t)) for k, r, t in zip(keys, reqs, tool_lists) if self.cache.get(k) is None]
        if todo:
            with ThreadPoolExecutor(self.workers) as ex:
                futs = {ex.submit(self._call, r, n): k for k, r, n in todo}
                for f in tqdm(as_completed(futs), total=len(futs), desc=desc):
                    self.cache.put(futs[f], f.result())
        return [self.cache.get(k) for k in keys]
