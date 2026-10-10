"""LangChain harness integration (design Sections 7.1, 7.3 and 7.6).

The agent starts with only the `find_tool` meta-tool. When it calls
find_tool(situation), the middleware runs the two-stage conformal selector and stores
the resulting set in agent state. `wrap_model_call` then exposes only find_tool plus the
exposed tools on every later model call, and `wrap_tool_call` lets only exposed tools
run and hosts an optional governance check (RQ5).

find_tool is a real tool whose effect is a state update, so no `after_model` jump is
needed (requirements.md, item 8 of section 4).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.types import Command
from typing_extensions import NotRequired

from . import conformal as cp
from .data import Catalog, Step, Tool
from .retrieval import Encoder, HybridPrefilter

FIND_TOOL_DESCRIPTION = (
    "Find tools for your next action. Describe the next action and its inputs and outputs, "
    "not the user's whole goal; for example 'send a Slack message to #ops with a summary', "
    "not 'help me with the incident'. Matching tools become available to call."
)


@tool("find_tool", description=FIND_TOOL_DESCRIPTION)
def find_tool(situation: str) -> str:
    """Placeholder body; ConformalToolExposureMiddleware intercepts every call."""
    return "find_tool is handled by ConformalToolExposureMiddleware."


class ExposureState(AgentState):
    exposed_tools: NotRequired[list[str]]


def _tool_name(t: BaseTool | dict) -> str:
    if isinstance(t, BaseTool):
        return t.name
    return t.get("name") or t.get("function", {}).get("name", "")


class ConformalSelector:
    """Prefilter -> Stage-2 source -> conformal set, over a live list of LangChain tools.

    `threshold` comes from offline calibration: q-hat for method "lac", tau for "crc".
    """

    def __init__(self, tools: list[BaseTool], source, threshold: float, k: int = 50, method: str = "lac",
                 k_max: int | None = None, encoder: Encoder | None = None):
        self.catalog = Catalog([Tool(t.name, t.name, t.description or "", tuple((t.args or {}).keys()))
                                for t in tools])
        self.prefilter = HybridPrefilter(self.catalog, encoder)
        self.source, self.threshold, self.method = source, threshold, method
        self.k, self.k_max = min(k, len(tools)), k_max

    def select(self, situation: str) -> dict[str, Any]:
        pool = self.prefilter.search([Step("live", situation, (), "live", "live")], self.k)
        probs = self.source.probs(pool)
        sets = cp.lac_sets(probs, self.threshold) if self.method == "lac" else cp.crc_sets(probs, self.threshold)
        empty = not sets.any()
        sets = cp.with_top1_fallback(sets, probs)
        raw_size = int(sets.sum())
        if self.k_max:
            sets = cp.cap_sets(sets, probs, self.k_max)
        members = np.flatnonzero(sets[0])
        members = members[np.argsort(-probs[0, members])]
        return {"situation": situation, "tools": [pool.tools[0][i].id for i in members],
                "probs": [float(probs[0, i]) for i in members], "empty": empty,
                "capped": int(sets.sum()) < raw_size}


class ConformalToolExposureMiddleware(AgentMiddleware):
    state_schema = ExposureState

    def __init__(self, selector: ConformalSelector, always_include: tuple[str, ...] = (),
                 governance: Callable[[dict], str | None] | None = None):
        super().__init__()
        self.selector = selector
        self.always = set(always_include)
        self.governance = governance
        self.tools = [find_tool]
        self.log: list[dict] = []

    def _exposed(self, state) -> set[str]:
        return {*self.always, *(state.get("exposed_tools") or [])}

    def wrap_model_call(self, request, handler):
        visible = {"find_tool"} | self._exposed(request.state)
        return handler(request.override(tools=[t for t in request.tools if _tool_name(t) in visible]))

    def wrap_tool_call(self, request, handler):
        call = request.tool_call
        if call["name"] == "find_tool":
            sel = self.selector.select(call["args"].get("situation", ""))
            self.log.append(sel)
            exposed = list(dict.fromkeys([*(request.state.get("exposed_tools") or []), *sel["tools"]]))
            descs = {t.id: t.description for t in self.selector.catalog.tools}
            content = "Tools now available:\n" + "\n".join(f"- {n}: {descs[n][:200]}" for n in sel["tools"])
            if sel["empty"]:
                content += "\n(No tool cleared the coverage threshold; showing the single best match. " \
                           "If it does not fit, call find_tool again with a more specific situation.)"
            msg = ToolMessage(content, tool_call_id=call["id"], name="find_tool")
            return Command(update={"exposed_tools": exposed, "messages": [msg]})
        if call["name"] not in self._exposed(request.state):
            return ToolMessage(f"Tool '{call['name']}' is not available. Call find_tool first.",
                               tool_call_id=call["id"], name=call["name"], status="error")
        if self.governance is not None and (reason := self.governance(call)):
            return ToolMessage(f"Blocked by governance check: {reason}", tool_call_id=call["id"],
                               name=call["name"], status="error")
        return handler(request)
