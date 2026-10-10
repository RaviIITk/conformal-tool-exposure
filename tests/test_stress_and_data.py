import numpy as np

from cte.data import Catalog, Tool, parse_tool
from cte.stress import INJECTION, drift, poison, sanitize, sanitize_catalog
from cte.toy_tools import SPECS


def toy_catalog():
    return Catalog([Tool(n, n, d, tuple(p), "web") for n, d, p in SPECS])


def test_sanitize_strips_injection_only():
    t = Tool("x", "weather_current", f"Get the current weather for a city. {INJECTION}")
    clean, flagged = sanitize(t)
    assert flagged and clean.description == "Get the current weather for a city."
    _, flagged = sanitize_catalog(toy_catalog())
    assert flagged == set()  # no false positives on ordinary descriptions


def test_poison_respects_protected_and_fraction():
    cat = toy_catalog()
    out, bad = poison(cat, 0.2, np.random.default_rng(0), protected={"slack_send_message"})
    assert len(bad) == 6 and "slack_send_message" not in bad
    assert all(INJECTION in t.description for t in out.tools if t.id in bad)


def test_sanity_check_restores_original_description_without_trailing_period():
    cat = Catalog([Tool("x", "file_character_count", "Count the number of characters in a file and return")])
    poisoned, _ = poison(cat, 1.0, np.random.default_rng(0), protected=set())
    clean, flagged = sanitize_catalog(poisoned)
    assert flagged == {"x"}
    assert clean.tools[0].description == "Count the number of characters in a file and return."


def test_drift_keeps_gold_and_adds_lookalikes():
    cat = toy_catalog()
    protected = {"slack_send_message", "email_send"}
    extra = [Tool(f"e{i}", f"extra_{i}", "Unrelated tool.") for i in range(20)]
    out = drift(cat, np.random.default_rng(0), protected, extra, near_dup=1.0)
    ids = {t.id for t in out.tools}
    assert protected <= ids
    assert {"dup_slack_send_message", "dup_email_send"} <= ids
    assert any(i.startswith("drift_add_") for i in ids)


def test_parse_tool_formats():
    web = parse_tool("w", '{"name": "file_character_count", "description": "Count characters", '
                          '"doc_arguments": {"type": "object", "properties": {"file_path": {"type": "string"}}}}', "web")
    assert web.params == ("file_path",)
    code = parse_tool("c", '{"name": "detector", "description": "Detect objects", "api_arguments": ["model_id"]}', "code")
    assert code.params == ("model_id",)
    rapid = parse_tool("r", '{"name": "Get_Order", "description": " ", "required_parameters": [{"name": "id"}], '
                            '"optional_parameters": []}', "web")
    assert rapid.text() == "Get_Order Parameters: id."
