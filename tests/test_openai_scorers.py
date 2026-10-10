"""OpenAI clients against a mocked HTTP endpoint, so the real SDK validates our requests."""

import json

import httpx
import numpy as np
import pytest
from openai import OpenAI

from cte.agent_eval import AgentProxy
from cte.data import Step, Tool
from cte.openai_scorers import DecisionsSource, LogprobSource
from cte.retrieval import Pool

TOOLS = [Tool(f"t{i}", f"tool_{i}", f"Does thing number {i}.", ("x",)) for i in range(4)]


def make_pool(n=3):
    steps = [Step(f"s{i}", f"do thing {i}", ("t1",), "x", "web") for i in range(n)]
    z = np.zeros((n, 4))
    return Pool(steps, [TOOLS] * n, z, z, z, np.tile([False, True, False, False], (n, 1)), np.ones(n, int))


def mock_client(handler, log):
    def route(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        log.append((request.url.path, body))
        return httpx.Response(200, json=handler(request.url.path, body))

    # Dummy key for the mock transport; no request leaves the process.
    return OpenAI(api_key="test-key", base_url="http://mock/v1", http_client=httpx.Client(transport=httpx.MockTransport(route)))


USAGE = {"input_tokens": 120, "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
         "output_tokens": 0, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 120}


def decisions_handler(path, body):
    assert path == "/v1/decisions"
    answers = []
    for q in body["questions"]:
        if q["type"] == "choice":
            values = [c["value"] for c in q["choices"]]
            probs = [0.7 if v == "tool_2" else 0.3 / (len(values) - 1) for v in values]
            answers.append({"type": "choice", "name": q["name"], "choice": "tool_2", "confidence": 0.7,
                            "probabilities": [{"value": v, "probability": p} for v, p in zip(values, probs)]})
        else:
            answers.append({"type": "predicate", "name": q["name"], "probability": 0.9 if q["name"] == "tool_2" else 0.1})
    return {"model": body["model"], "answers": answers, "usage": USAGE}


def test_decisions_choice(tmp_path):
    log = []
    src = DecisionsSource("choice", tmp_path, model="gpt-6-luna", client=mock_client(decisions_handler, log))
    probs = src.probs(make_pool())
    assert probs.shape == (3, 4)
    assert np.allclose(probs[:, 1], 0.7)
    assert probs.sum(1) == pytest.approx(0.7 + 3 * 0.3 / 4)  # none_of_these keeps its mass
    req = log[0][1]
    assert req["questions"][0]["choices"][-1]["value"] == "none_of_these"
    assert len(req["questions"][0]["choices"]) == 5
    # Cached: a second pass makes no new calls.
    src.probs(make_pool())
    assert len(log) == 3
    assert src.usage(make_pool())["input_tokens"] == 360


def test_decisions_predicate_with_risk_questions(tmp_path):
    log = []
    src = DecisionsSource("predicate", tmp_path, client=mock_client(decisions_handler, log),
                          risk_questions={"risk_privacy": "Does this expose private data?"})
    probs = src.probs(make_pool(1))
    assert probs.tolist() == [[0.1, 0.9, 0.1, 0.1]]
    assert len(log[0][1]["questions"]) == 5
    assert src.records(make_pool(1))[0]["extra"] == {"risk_privacy": 0.1}


def test_decisions_refusal(tmp_path):
    def refuse(path, body):
        return {"model": body["model"], "answers": [{"type": "refusal", "name": "tool"}], "usage": USAGE}

    src = DecisionsSource("choice", tmp_path, client=mock_client(refuse, []))
    assert src.probs(make_pool(1)).tolist() == [[0, 0, 0, 0]]
    assert src.usage(make_pool(1))["refusals"] == 1


def systemone_handler(path, body):
    """Jev-compatible response: answers keyed by question name."""
    assert path == "/api/alpha/decisions" and isinstance(body["questions"], dict)
    answers = {}
    for name, q in body["questions"].items():
        if q["type"] == "choice":
            keys = list(q["criteria"])
            answers[name] = {"type": "choice", "choice": "tool_2",
                             "probabilities": {k: (0.6 if k == "tool_2" else 0.4 / (len(keys) - 1)) for k in keys}}
        else:
            answers[name] = {"type": "noul", "noul": 0.8 if name == "tool_2" else 0.05}
    return {"model": body["model"], "answers": answers, "usage": {"input_tokens": 90, "output_tokens": 0}}


@pytest.fixture
def openrouter_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")  # dummy value for the mock transport


@pytest.mark.parametrize("mode,expected", [("choice", [0.1, 0.6, 0.1, 0.1]), ("predicate", [0.05, 0.8, 0.05, 0.05])])
def test_systemone_source(tmp_path, openrouter_env, mode, expected):
    log = []

    def route(request):
        log.append(request)
        return httpx.Response(200, json=systemone_handler(request.url.path, json.loads(request.content)))

    from cte.openai_scorers import SystemOneSource

    src = SystemOneSource(mode, tmp_path, http_client=httpx.Client(transport=httpx.MockTransport(route)))
    assert np.allclose(src.probs(make_pool(2)), [expected] * 2)
    body = json.loads(log[0].content)
    assert body["model"] == "inception/mercury-decide:free" and body["state"] == "do thing 0"
    assert log[0].headers["authorization"] == "Bearer test-key"
    if mode == "choice":
        assert list(body["questions"]["tool"]["criteria"])[-1] == "none_of_these"
    else:
        assert len(body["questions"]) == 4


def test_systemone_retries_on_429(tmp_path, openrouter_env, monkeypatch):
    import cte.openai_scorers as mod

    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def route(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "1"}, json={"error": "rate limited"})
        return httpx.Response(200, json=systemone_handler(request.url.path, json.loads(request.content)))

    src = mod.SystemOneSource("choice", tmp_path, http_client=httpx.Client(transport=httpx.MockTransport(route)))
    assert src.probs(make_pool(1))[0, 1] == pytest.approx(0.6)
    assert calls["n"] == 2


def chat_response(body, content=None, tool_calls=None, logprobs=None):
    return {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "logprobs": logprobs,
                         "message": {"role": "assistant", "content": content, "tool_calls": tool_calls}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 1, "total_tokens": 51}}


def test_logprob_source(tmp_path):
    def handler(path, body):
        assert path == "/v1/chat/completions" and body["top_logprobs"] == 20
        top = [{"token": "B", "logprob": np.log(0.6), "bytes": None},
               {"token": " A", "logprob": np.log(0.2), "bytes": None},
               {"token": "Z", "logprob": np.log(0.1), "bytes": None}]
        lp = {"content": [{"token": "B", "logprob": np.log(0.6), "bytes": None, "top_logprobs": top}]}
        return chat_response(body, content="B", logprobs=lp)

    src = LogprobSource(tmp_path, model="gpt-4.1-mini", client=mock_client(handler, []))
    assert np.allclose(src.probs(make_pool(1)), [[0.25, 0.75, 0, 0]])


def test_agent_proxy(tmp_path):
    def handler(path, body):
        name = body["tools"][1]["function"]["name"]
        return chat_response(body, tool_calls=[{"id": "c", "type": "function",
                                                "function": {"name": name, "arguments": "{}"}}])

    agent = AgentProxy(tmp_path, model="gpt-5-mini", client=mock_client(handler, []))
    recs = agent.run(["do thing"], [TOOLS[:3]])
    assert recs[0]["chosen"] == 1 and recs[0]["prompt_tokens"] == 50
