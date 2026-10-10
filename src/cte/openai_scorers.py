"""Stage-2 score sources backed by hosted decision models and LLM APIs.

- SystemOneSource: any Jev-compatible decision model (`/v1/systemone` schema); defaults to
  Inception Mercury Decide on OpenRouter's Decisions API, the decision model in this study.
- DecisionsSource: the OpenAI Decisions API (`POST /v1/decisions`, model gpt-6-luna).
  Both ask Mode A as one choice over the K candidates plus `none_of_these`, and Mode B as
  one yes/no question per candidate.
- LogprobSource: Prune 'n Predict-style LLM scores, i.e. next-token probabilities of the
  option letters from Chat Completions with logprobs.

Every request is cached in a JSONL file keyed by its exact body, so reruns, the CP
experiments and the stress tests never pay twice for the same call.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import numpy as np
from tqdm import tqdm

from .cache import JsonlCache, text_key
from .data import Tool
from .retrieval import Pool

DECISIONS_PRICE_PER_M_INPUT = 0.10  # USD, input tokens only (OpenAI pricing, Oct 2026)
OPENROUTER_BASE = "https://openrouter.ai/api"

CHOICE_INSTRUCTIONS = (
    "The input describes the next action an AI agent must take. "
    "Which tool should the agent call to perform exactly this action?"
)
PREDICATE_INSTRUCTIONS = "Tool: {tool}\n\nCan this tool perform the action described in the input?"
NONE_CHOICE = {"value": "none_of_these", "description": "None of the listed tools can perform this action."}
LETTERS = [chr(ord("A") + i) for i in range(26)]


def make_client(openrouter_ok: bool = True):
    """OpenAI SDK client: api.openai.com if OPENAI_API_KEY is set, else OpenRouter's compatible API."""
    from openai import OpenAI

    if os.environ.get("OPENAI_API_KEY"):
        return OpenAI(max_retries=6, timeout=120)
    if openrouter_ok and os.environ.get("OPENROUTER_API_KEY"):
        return OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=f"{OPENROUTER_BASE}/v1",
                      max_retries=6, timeout=120)
    raise RuntimeError("Set OPENAI_API_KEY (or OPENROUTER_API_KEY for chat models) in .env; see .env.example.")


class _ApiSource:
    """Shared plumbing: build request -> cached, parallel API calls -> parse to probabilities."""

    name: str
    mode: str

    def __init__(self, cache_path: Path, client=None, workers: int = 8, max_chars: int = 300):
        self.cache = JsonlCache(cache_path)
        self._client = client
        self.workers, self.max_chars = workers, max_chars

    @property
    def client(self):
        if self._client is None:
            self._client = make_client()
        return self._client

    def fit(self, dev: Pool) -> None:
        """API probabilities are used as returned (no post-hoc calibration)."""

    def _request(self, situation: str, tools: list[Tool]) -> dict:
        raise NotImplementedError

    def _call(self, req: dict) -> dict:
        raise NotImplementedError

    def _parse(self, resp: dict, k: int) -> tuple[list[float], bool, dict]:
        raise NotImplementedError

    def _input_tokens(self, resp: dict) -> int:
        raise NotImplementedError

    def records(self, pool: Pool) -> list[dict]:
        reqs = [self._request(s.situation, tools) for s, tools in zip(pool.steps, pool.tools)]
        keys = [text_key(json.dumps(r, sort_keys=True)) for r in reqs]
        todo = {k: r for k, r in zip(keys, reqs) if self.cache.get(k) is None}
        budget = os.environ.get("CTE_MAX_API_CALLS")
        if budget is not None and len(todo) > int(budget):
            raise RuntimeError(f"{self.name}: {len(todo)} uncached calls needed but CTE_MAX_API_CALLS={budget}")

        def run(key: str, req: dict) -> None:
            t0 = time.perf_counter()
            resp = self._call(req)
            # Sources with client-side rate limiting report pure API time, excluding queueing.
            latency = resp.pop("_latency", None) or time.perf_counter() - t0
            probs, refused, extra = self._parse(resp, pool.k)
            self.cache.put(key, {"probs": probs, "refused": refused, "extra": extra, "latency": latency,
                                 "input_tokens": self._input_tokens(resp), "model": resp.get("model")})

        if todo:
            with ThreadPoolExecutor(self.workers) as ex:
                futures = [ex.submit(run, k, r) for k, r in todo.items()]
                for f in tqdm(as_completed(futures), total=len(futures), desc=self.name):
                    f.result()
        return [self.cache.get(k) for k in keys]

    def probs(self, pool: Pool) -> np.ndarray:
        return np.array([r["probs"] for r in self.records(pool)], dtype=np.float64)

    def usage(self, pool: Pool) -> dict:
        recs = self.records(pool)
        tokens = sum(r["input_tokens"] for r in recs)
        lat = np.array([r["latency"] for r in recs])
        return {"calls": len(recs), "input_tokens": tokens, "refusals": sum(r["refused"] for r in recs),
                "latency_median_s": float(np.median(lat)), "latency_p95_s": float(np.percentile(lat, 95))}


class DecisionsSource(_ApiSource):
    """OpenAI Decisions API as the Stage-2 decision model.

    `risk_questions` (name -> instructions) adds governance predicates to the same call (RQ5).
    """

    def __init__(self, mode: str, cache_dir: Path, model: str | None = None,
                 risk_questions: dict[str, str] | None = None, **kw):
        self.mode = mode
        self.model = model or os.environ.get("CTE_DECISIONS_MODEL", "gpt-6-luna")
        self.risk_questions = risk_questions or {}
        self.name = f"decisions-{mode}" + ("+risk" if self.risk_questions else "")
        super().__init__(cache_dir / f"openai_{self.model}_{self.name}.jsonl", **kw)

    def _request(self, situation: str, tools: list[Tool]) -> dict:
        if self.mode == "choice":
            choices = [{"value": f"tool_{i + 1}", "description": t.text(self.max_chars)} for i, t in enumerate(tools)]
            questions = [{"type": "choice", "name": "tool", "instructions": CHOICE_INSTRUCTIONS,
                          "choices": choices + [NONE_CHOICE]}]
        else:
            questions = [{"type": "predicate", "name": f"tool_{i + 1}",
                          "instructions": PREDICATE_INSTRUCTIONS.format(tool=t.text(self.max_chars))}
                         for i, t in enumerate(tools)]
        questions += [{"type": "predicate", "name": n, "instructions": q} for n, q in self.risk_questions.items()]
        return {"model": self.model, "input": situation, "questions": questions}

    @property
    def client(self):
        if self._client is None:
            self._client = make_client(openrouter_ok=False)  # /v1/decisions is OpenAI-only
        return self._client

    def _call(self, req: dict) -> dict:
        return self.client.decisions.create(**req).model_dump()

    def _input_tokens(self, resp: dict) -> int:
        return int((resp.get("usage") or {}).get("input_tokens", 0))

    def _parse(self, resp: dict, k: int) -> tuple[list[float], bool, dict]:
        answers = resp["answers"]
        if all(a.get("name") is None for a in answers):  # fall back to request order
            names = (["tool"] if self.mode == "choice" else [f"tool_{i + 1}" for i in range(k)]) + list(self.risk_questions)
            answers = [{**a, "name": n} for a, n in zip(answers, names)]
        by_name = {a["name"]: a for a in answers}
        refused = False
        if self.mode == "choice":
            a = by_name.get("tool", {"type": "refusal"})
            refused = a["type"] == "refusal"
            dist = {} if refused else {p["value"]: p["probability"] for p in a["probabilities"]}
            probs = [float(dist.get(f"tool_{i + 1}", 0.0)) for i in range(k)]
        else:
            probs = []
            for i in range(k):
                a = by_name.get(f"tool_{i + 1}", {"type": "refusal"})
                refused |= a["type"] == "refusal"
                probs.append(float(a.get("probability", 0.0)))
        extra = {n: by_name.get(n, {}).get("probability") for n in self.risk_questions}
        return probs, refused, extra


class RateLimiter:
    """At most `rpm` calls in any 60-second window, shared by all worker threads."""

    def __init__(self, rpm: int):
        self.rpm = rpm
        self._lock = threading.Lock()
        self._times: deque[float] = deque()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._times and now - self._times[0] >= 60:
                    self._times.popleft()
                if len(self._times) < self.rpm:
                    self._times.append(now)
                    return
                wait = 60 - (now - self._times[0])
            time.sleep(max(wait, 0.05))


def openrouter_quota(api_key: str | None = None, http_client: httpx.Client | None = None) -> dict:
    """Free-model daily request counter for this key ({used, limit, remaining}); {} if unavailable."""
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    client = http_client or httpx.Client(timeout=30)
    resp = client.get(f"{OPENROUTER_BASE}/v1/key", headers={"Authorization": f"Bearer {key}"})
    resp.raise_for_status()
    body = resp.json()
    return (body.get("data") or body).get("free_model_daily_requests") or {}


class SystemOneSource(_ApiSource):
    """Jev-compatible decision model over the `/v1/systemone` schema.

    Defaults to OpenRouter's Decisions API with Inception Mercury Decide (free tier:
    20 requests/min; 50/day, or 1,000/day after $10 of credits). The same request format
    serves TypeSafe Jev and Cloudflare Clef by changing `url`, `model` and `api_key_env`.
    Mode A asks one `choice` over the K candidates plus `none_of_these`; Mode B asks one
    `noul` (yes/no) question per candidate. One request scores one step in either mode.
    """

    def __init__(self, mode: str, cache_dir: Path, model: str | None = None, url: str | None = None,
                 api_key_env: str = "OPENROUTER_API_KEY", rpm: int = 20, max_retries: int = 8,
                 risk_questions: dict[str, str] | None = None, http_client: httpx.Client | None = None, **kw):
        self.mode = mode
        self.model = model or os.environ.get("CTE_SYSTEMONE_MODEL", "inception/mercury-decide:free")
        self.url = url or os.environ.get("CTE_SYSTEMONE_URL", f"{OPENROUTER_BASE}/alpha/decisions")
        self.api_key_env, self.max_retries = api_key_env, max_retries
        self.risk_questions = risk_questions or {}
        self.limiter = RateLimiter(rpm)
        self.http = http_client or httpx.Client(timeout=120)
        self.name = f"{self.model.split('/')[-1].split(':')[0]}-{mode}" + ("+risk" if self.risk_questions else "")
        safe_model = self.model.replace("/", "__").replace(":", "_")
        super().__init__(cache_dir / f"systemone_{safe_model}_{mode}{'_risk' if self.risk_questions else ''}.jsonl",
                         client=self.http, **kw)

    def _request(self, situation: str, tools: list[Tool]) -> dict:
        if self.mode == "choice":
            criteria = {f"tool_{i + 1}": t.text(self.max_chars) for i, t in enumerate(tools)}
            criteria[NONE_CHOICE["value"]] = NONE_CHOICE["description"]
            questions = {"tool": {"type": "choice", "instructions": CHOICE_INSTRUCTIONS, "criteria": criteria}}
        else:
            questions = {f"tool_{i + 1}": {
                "type": "noul", "instructions": PREDICATE_INSTRUCTIONS.format(tool=t.text(self.max_chars)),
                "criteria": {"true": "The tool can perform this action.",
                             "false": "The tool cannot perform this action."}} for i, t in enumerate(tools)}
        for name, q in self.risk_questions.items():
            questions[name] = {"type": "noul", "instructions": q}
        return {"model": self.model, "state": situation, "questions": questions}

    def _call(self, req: dict) -> dict:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"{self.api_key_env} is not set. Put it in .env (see .env.example).")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                   "X-OpenRouter-Title": "conformal-tool-exposure"}
        for attempt in range(self.max_retries):
            self.limiter.acquire()
            t0 = time.perf_counter()
            resp = self.http.post(self.url, headers=headers, json=req)
            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = resp.headers.get("retry-after")
                time.sleep(float(retry_after) if retry_after else min(60, 5 * 2 ** attempt))
                continue
            if resp.status_code >= 400:
                raise RuntimeError(f"decision API error {resp.status_code}: {resp.text[:500]}")
            return {**resp.json(), "_latency": time.perf_counter() - t0}
        raise RuntimeError(f"decision API still rate-limited after {self.max_retries} retries")

    def _input_tokens(self, resp: dict) -> int:
        usage = resp.get("usage") or {}
        return int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)

    def _parse(self, resp: dict, k: int) -> tuple[list[float], bool, dict]:
        answers = resp.get("answers") or {}
        if isinstance(answers, list):
            answers = {a.get("name"): a for a in answers}
        refused = False
        if self.mode == "choice":
            a = answers.get("tool") or {}
            dist = a.get("probabilities")
            if isinstance(dist, list):
                dist = {d.get("value", d.get("option")): d.get("probability") for d in dist}
            refused = not dist
            probs = [float((dist or {}).get(f"tool_{i + 1}", 0.0)) for i in range(k)]
        else:
            probs = []
            for i in range(k):
                a = answers.get(f"tool_{i + 1}") or {}
                p = a.get("noul", a.get("probability"))
                refused |= p is None
                probs.append(float(p or 0.0))
        extra = {n: (answers.get(n) or {}).get("noul") for n in self.risk_questions}
        return probs, refused, extra


class LogprobSource(_ApiSource):
    """LLM log-probability baseline: P(option letter) from Chat Completions, renormalised."""

    mode = "choice"

    def __init__(self, cache_dir: Path, model: str | None = None, **kw):
        self.model = model or os.environ.get("CTE_LOGPROB_MODEL", "gpt-4.1-mini")
        self.name = "llm-logprob"
        super().__init__(cache_dir / f"openai_{self.model}_logprob.jsonl", **kw)

    # Chat Completions returns at most 20 top_logprobs, so only 20 letters are observable.
    max_k = 20

    def records(self, pool: Pool) -> list[dict]:
        return super().records(pool.truncate(min(pool.k, self.max_k)))

    def probs(self, pool: Pool) -> np.ndarray:
        p = super().probs(pool)
        return np.pad(p, ((0, 0), (0, pool.k - p.shape[1])))

    def _request(self, situation: str, tools: list[Tool]) -> dict:
        listing = "\n".join(f"{LETTERS[i]}. {t.text(self.max_chars)}" for i, t in enumerate(tools))
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You choose tools for an AI agent. Reply with one letter only."},
                {"role": "user", "content": f"Next action: {situation}\n\nTools:\n{listing}\n\n"
                                            "Which tool performs this action? Answer with its letter."},
            ],
            "max_completion_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20,
        }

    def _call(self, req: dict) -> dict:
        return self.client.chat.completions.create(**req).model_dump()

    def _input_tokens(self, resp: dict) -> int:
        return int((resp.get("usage") or {}).get("prompt_tokens", 0))

    def _parse(self, resp: dict, k: int) -> tuple[list[float], bool, dict]:
        probs = np.zeros(k)
        content = (resp["choices"][0].get("logprobs") or {}).get("content") or []
        for entry in (content[0]["top_logprobs"] if content else []):
            tok = entry["token"].strip().rstrip(".")
            if tok in LETTERS[:k]:
                probs[LETTERS.index(tok)] += math.exp(entry["logprob"])
        total = float(probs.sum())
        return (probs / total if total > 0 else probs).tolist(), total == 0, {}
