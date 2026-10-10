"""A small multi-domain tool catalog and a scripted agent, for the middleware demo and tests."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import Field, create_model

SPECS = [
    ("slack_send_message", "Send a message to a Slack channel.", ["channel", "text"]),
    ("slack_list_channels", "List the Slack channels in the workspace.", []),
    ("slack_search_messages", "Search Slack messages by keyword.", ["query"]),
    ("email_send", "Send an email to a recipient with a subject and body.", ["to", "subject", "body"]),
    ("email_search", "Search the user's mailbox for emails matching a query.", ["query"]),
    ("calendar_create_event", "Create a calendar event with a title, start time and attendees.",
     ["title", "start", "attendees"]),
    ("calendar_list_events", "List calendar events in a date range.", ["start", "end"]),
    ("weather_current", "Get the current weather conditions for a city.", ["city"]),
    ("weather_forecast", "Get a multi-day weather forecast for a city.", ["city", "days"]),
    ("github_create_issue", "Open a new issue in a GitHub repository.", ["repo", "title", "body"]),
    ("github_list_pull_requests", "List open pull requests in a GitHub repository.", ["repo"]),
    ("github_merge_pull_request", "Merge a pull request in a GitHub repository.", ["repo", "number"]),
    ("jira_create_ticket", "Create a Jira ticket in a project.", ["project", "summary"]),
    ("jira_update_status", "Change the status of a Jira ticket.", ["ticket", "status"]),
    ("file_read", "Read the contents of a local file.", ["path"]),
    ("file_write", "Write text to a local file, overwriting it.", ["path", "text"]),
    ("file_delete", "Delete a local file permanently.", ["path"]),
    ("db_query", "Run a read-only SQL query against the analytics database.", ["sql"]),
    ("db_execute", "Execute a SQL statement that modifies the database.", ["sql"]),
    ("payments_refund", "Refund a customer payment by charge id.", ["charge_id", "amount"]),
    ("payments_list_charges", "List recent payment charges for a customer.", ["customer_id"]),
    ("crm_lookup_customer", "Look up a customer record in the CRM by email address.", ["email"]),
    ("crm_update_customer", "Update fields on a CRM customer record.", ["customer_id", "fields"]),
    ("translate_text", "Translate text into a target language.", ["text", "language"]),
    ("summarize_text", "Summarize a long text into a few sentences.", ["text"]),
    ("web_search", "Search the web and return the top results.", ["query"]),
    ("web_fetch", "Fetch the contents of a web page by URL.", ["url"]),
    ("maps_directions", "Get driving directions between two places.", ["origin", "destination"]),
    ("stock_quote", "Get the latest stock price for a ticker symbol.", ["ticker"]),
    ("currency_convert", "Convert an amount between two currencies.", ["amount", "from_currency", "to_currency"]),
]


def make_toy_tools(calls: list) -> list[BaseTool]:
    """Build the catalog; every executed tool appends (name, args) to `calls`."""

    def make(name: str):
        def run(**kwargs: Any) -> str:
            calls.append((name, kwargs))
            return f"{name}: done"

        return run

    return [StructuredTool.from_function(make(name), name=name, description=desc,
                                         args_schema=create_model(f"{name}_args", **{p: (str, ...) for p in params}))
            for name, desc, params in SPECS]


def _name(t) -> str:
    return t.name if isinstance(t, BaseTool) else t.get("name") or t.get("function", {}).get("name", "")


class ScriptedToolModel(BaseChatModel):
    """Deterministic stand-in agent: call find_tool, then the target tool if it is visible, then stop."""

    situation: str
    target: str
    target_args: dict = Field(default_factory=dict)
    bound: list[str] = Field(default_factory=list)
    seen: list = Field(default_factory=list)  # tool names the model could see on each call (shared by copies)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self.model_copy(update={"bound": [_name(t) for t in tools]})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append(list(self.bound))
        n_results = sum(isinstance(m, ToolMessage) for m in messages)
        if n_results == 0:
            msg = AIMessage("", tool_calls=[{"name": "find_tool", "args": {"situation": self.situation},
                                             "id": "call_find", "type": "tool_call"}])
        elif n_results == 1 and self.target in self.bound:
            msg = AIMessage("", tool_calls=[{"name": self.target, "args": self.target_args,
                                             "id": "call_target", "type": "tool_call"}])
        elif n_results == 1:
            msg = AIMessage(f"I could not find {self.target}.")
        else:
            msg = AIMessage("Done.")
        return ChatResult(generations=[ChatGeneration(message=msg)])
