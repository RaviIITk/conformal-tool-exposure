"""ToolRet loading: one tool catalog per category, and labelled agent steps.

ToolRet (arXiv 2503.01763) has 7,961 queries over 44,453 tools in three categories
(web, code, customized). Each query is one agent step: a situation plus the gold
tools that step needs. Retrieval for a step runs over its category's catalog.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

TOOLRET_URL = "https://huggingface.co/datasets/mangopy/{repo}/resolve/main/{path}"
CATEGORIES = ("web", "code", "customized")
SUBSETS = (
    "apibank", "apigen", "appbench", "autotools-food", "autotools-music",
    "autotools-weather", "craft-math-algebra", "craft-tabmwp", "craft-vqa",
    "gorilla-huggingface", "gorilla-pytorch", "gorilla-tensor", "gpt4tools", "gta",
    "metatool", "mnms", "restgpt-spotify", "restgpt-tmdb", "reversechain", "rotbench",
    "t-eval-dialog", "t-eval-step", "taskbench-daily", "taskbench-huggingface",
    "taskbench-multimedia", "tool-be-honest", "toolace", "toolalpaca", "toolbench-sam",
    "toolbench", "toolemu", "tooleyes", "toolink", "toollens", "ultratool",
)

_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class Tool:
    id: str
    name: str
    description: str
    params: tuple[str, ...] = ()
    category: str = ""

    def text(self, max_chars: int = 400) -> str:
        """One-line rendering used by every retriever and scorer."""
        out = f"{self.name}: {self.description}" if self.description else self.name
        if self.params:
            out += f" Parameters: {', '.join(self.params[:8])}."
        return out if len(out) <= max_chars else out[: max_chars - 3] + "..."


@dataclass
class Step:
    """One agent step: what the agent needs to do next, and the gold tools for it."""

    id: str
    situation: str
    gold: tuple[str, ...]
    subset: str
    category: str


@dataclass
class Catalog:
    tools: list[Tool]
    index: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.index = {t.id: i for i, t in enumerate(self.tools)}

    def __len__(self) -> int:
        return len(self.tools)


def download_toolret(root: Path) -> None:
    """Fetch ToolRet parquet files into root/{tools,queries} unless already present."""
    jobs = [("ToolRet-Tools", f"{c}/tools-00000-of-00001.parquet", root / "tools" / f"{c}.parquet")
            for c in CATEGORIES]
    jobs += [("ToolRet-Queries", f"{s}/queries-00000-of-00001.parquet", root / "queries" / f"{s}.parquet")
             for s in SUBSETS]
    for repo, path, dest in jobs:
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TOOLRET_URL.format(repo=repo, path=path), dest)


def _clean(text) -> str:
    return _WS.sub(" ", str(text or "")).strip()


def _param_names(doc: dict) -> tuple[str, ...]:
    names: list[str] = []
    for key in ("parameters", "doc_arguments"):
        p = doc.get(key)
        if isinstance(p, dict):
            props = p.get("properties", {} if "type" in p else p)
            if isinstance(props, dict):
                names += list(props)
        elif isinstance(p, list):
            names += [x["name"] for x in p if isinstance(x, dict) and "name" in x]
    for key in ("required_parameters", "optional_parameters"):
        p = doc.get(key)
        if isinstance(p, list):
            names += [x["name"] for x in p if isinstance(x, dict) and "name" in x]
    args = doc.get("api_arguments")
    if isinstance(args, list):
        names += [str(x) for x in args]
    elif isinstance(args, dict):
        names += list(args)
    seen: dict[str, None] = {}
    for n in names:
        if isinstance(n, str) and n:
            seen.setdefault(n, None)
    return tuple(seen)


def parse_tool(tool_id: str, documentation: str, category: str) -> Tool:
    doc = json.loads(documentation)
    name = _clean(doc.get("name") or doc.get("api_name") or tool_id)
    desc = doc.get("description") or doc.get("func_description") or doc.get("functionality") or ""
    if not isinstance(desc, str):
        desc = json.dumps(desc)
    return Tool(tool_id, name, _clean(desc), _param_names(doc), category)


def load_catalogs(root: Path) -> dict[str, Catalog]:
    out = {}
    for c in CATEGORIES:
        df = pd.read_parquet(root / "tools" / f"{c}.parquet")
        out[c] = Catalog([parse_tool(i, d, c) for i, d in zip(df["id"], df["documentation"])])
    return out


def load_steps(root: Path, catalogs: dict[str, Catalog]) -> list[Step]:
    """Load every ToolRet query whose gold tools all exist in its category catalog."""
    steps = []
    for s in SUBSETS:
        df = pd.read_parquet(root / "queries" / f"{s}.parquet")
        for row in df.itertuples(index=False):
            gold = tuple(dict.fromkeys(l["id"] for l in json.loads(row.labels) if l.get("relevance", 1) > 0))
            cat = catalogs[row.category]
            if gold and all(g in cat.index for g in gold):
                steps.append(Step(row.id, _clean(row.query), gold, s, row.category))
    return steps


def load_toolret(root: Path) -> tuple[dict[str, Catalog], list[Step]]:
    download_toolret(root)
    catalogs = load_catalogs(root)
    return catalogs, load_steps(root, catalogs)
