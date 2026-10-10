"""End-to-end LangChain agent run with the conformal exposure middleware and a scripted model."""

from types import SimpleNamespace

import pytest
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from cte.middleware import ConformalSelector, ConformalToolExposureMiddleware
from cte.scorers import CalibratedSource, feature_logits
from cte.toy_tools import SPECS, ScriptedToolModel, make_toy_tools


def build(calls, governance=None, threshold=0.9):
    tools = make_toy_tools(calls)
    src = CalibratedSource("bm25", feature_logits("bm25"), "choice")
    src.params = (1.0,)
    selector = ConformalSelector(tools, src, threshold=threshold, k=10)  # BM25-only prefilter
    return tools, ConformalToolExposureMiddleware(selector, governance=governance)


def run(model, tools, mw):
    agent = create_agent(model=model, tools=tools, middleware=[mw])
    return agent.invoke({"messages": [HumanMessage("Tell the ops channel the deploy finished.")]})


def test_agent_sees_only_find_tool_then_the_conformal_set():
    calls = []
    tools, mw = build(calls)
    model = ScriptedToolModel(situation="send a message to the ops Slack channel", target="slack_send_message",
                              target_args={"channel": "#ops", "text": "deploy finished"})
    state = run(model, tools, mw)

    assert model.seen[0] == ["find_tool"]
    exposed = mw.log[0]["tools"]
    assert "slack_send_message" in exposed
    assert 1 <= len(exposed) < len(SPECS)
    assert sorted(model.seen[1]) == sorted(["find_tool", *exposed])
    assert calls == [("slack_send_message", {"channel": "#ops", "text": "deploy finished"})]
    assert state["exposed_tools"] == exposed
    assert state["messages"][-1].content == "Done."


def test_tool_outside_the_set_is_not_visible():
    calls = []
    tools, mw = build(calls)
    model = ScriptedToolModel(situation="send a message to the ops Slack channel", target="db_execute")
    state = run(model, tools, mw)
    assert "db_execute" not in model.seen[1]
    assert calls == []
    assert "could not find" in state["messages"][-1].content


def test_wrap_tool_call_blocks_unexposed_and_governed_calls():
    calls = []
    _, mw = build(calls, governance=lambda call: "destructive action" if call["name"] == "file_delete" else None)

    def handler(_):
        pytest.fail("tool should not run")

    req = SimpleNamespace(tool_call={"name": "db_execute", "args": {}, "id": "1"}, state={"exposed_tools": []})
    msg = mw.wrap_tool_call(req, handler)
    assert msg.status == "error" and "find_tool" in msg.content

    req = SimpleNamespace(tool_call={"name": "file_delete", "args": {}, "id": "2"},
                          state={"exposed_tools": ["file_delete"]})
    msg = mw.wrap_tool_call(req, handler)
    assert msg.status == "error" and "destructive" in msg.content


def test_empty_set_falls_back_to_top1():
    calls = []
    _, mw = build(calls, threshold=-1.0)  # q-hat below every score: the raw set is empty
    sel = mw.selector.select("refund the customer's payment")
    assert sel["empty"] and len(sel["tools"]) == 1
