"""Demo: a LangChain agent that only ever sees a conformal set of tools.

The score source and its calibration (temperature, q-hat) come from a Phase 1 run.
`--source mercury-choice` uses the OpenRouter decision model (OPENROUTER_API_KEY).
The agent is scripted unless OPENAI_API_KEY is set, in which case CTE_AGENT_MODEL runs it.

  python scripts/demo_agent.py --phase1 results/minilm/phase1/results.json --source hybrid
  python scripts/demo_agent.py --phase1 results/mercury_smoke/phase1/results.json --source mercury-choice
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from cte.middleware import ConformalSelector, ConformalToolExposureMiddleware
from cte.pipeline import make_source
from cte.retrieval import Encoder
from cte.toy_tools import ScriptedToolModel, make_toy_tools

TASKS = [
    ("Tell the #ops Slack channel that the deploy finished.", "send a message to the ops Slack channel",
     "slack_send_message", {"channel": "#ops", "text": "Deploy finished."}),
    ("What's the weather in Paris right now?", "get the current weather for Paris", "weather_current",
     {"city": "Paris"}),
    ("Refund charge ch_123 for $20.", "refund the customer payment ch_123 of 20 dollars", "payments_refund",
     {"charge_id": "ch_123", "amount": "20"}),
    ("Delete /tmp/old.log.", "delete the local file /tmp/old.log", "file_delete", {"path": "/tmp/old.log"}),
]
GUARDED = {"file_delete", "db_execute", "github_merge_pull_request"}


def governance(call: dict) -> str | None:
    """RQ5 hook: hold destructive tools for human approval."""
    return "destructive action needs human approval" if call["name"] in GUARDED else None


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase1", type=Path, default=Path("results/minilm/phase1/results.json"))
    p.add_argument("--source", default="hybrid",
                   help="local: bm25|dense|hybrid|ce; API: mercury-choice|decisions-choice")
    p.add_argument("--alpha", default="0.1")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--cache", type=Path, default=Path("cache"))
    args = p.parse_args()

    live = bool(os.environ.get("OPENAI_API_KEY"))
    phase1 = json.loads(args.phase1.read_text())
    source_name = args.source
    saved = phase1["sources"].get(source_name)
    src = make_source(source_name, SimpleNamespace(ce_model="cross-encoder/ms-marco-MiniLM-L6-v2", cache=args.cache))
    if saved:
        src.params = tuple(saved["params"])
        threshold = saved["thresholds"][f"lac|{args.alpha}"]
    else:
        threshold = 0.9
        print(f"[warn] no Phase 1 calibration for {source_name}; using q-hat = {threshold}")
    encoder = Encoder(phase1["encoder"], args.cache) if source_name != "bm25" else None
    print(f"source={source_name} q-hat={threshold:.4f} agent={'OpenAI ' + os.environ.get('CTE_AGENT_MODEL', 'gpt-5-mini') if live else 'scripted'}")

    for request, situation, target, target_args in TASKS:
        calls: list = []
        tools = make_toy_tools(calls)
        mw = ConformalToolExposureMiddleware(ConformalSelector(tools, src, threshold, k=args.k, encoder=encoder),
                                             governance=governance)
        if live:
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(model=os.environ.get("CTE_AGENT_MODEL", "gpt-5-mini"))
        else:
            model = ScriptedToolModel(situation=situation, target=target, target_args=target_args)
        state = create_agent(model=model, tools=tools, middleware=[mw]).invoke({"messages": [HumanMessage(request)]})
        print(f"\n> {request}")
        for sel in mw.log:
            shown = ", ".join(f"{n} ({p:.2f})" for n, p in zip(sel["tools"], sel["probs"]))
            print(f"  find_tool({sel['situation']!r}) -> {shown}" + ("  [empty set: top-1 fallback]" if sel["empty"] else ""))
        print(f"  executed: {calls or 'nothing'}")
        blocked = [m.content for m in state["messages"] if getattr(m, "status", None) == "error"]
        if blocked:
            print(f"  blocked: {blocked}")
        print(f"  agent: {state['messages'][-1].content}")


if __name__ == "__main__":
    main()
