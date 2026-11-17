"""Controlled web search.

`search_tools(mode)` is the one switch every task uses:

  off         no search tool at all (the model cannot search)
  native      Inspect's web_search() with the model provider's own server-side search
              (OpenAI / Anthropic / Gemini / Grok / Perplexity). Quality differs by vendor.
  tavily      Inspect's stock web_search() backed by Tavily (same backend for every model).
  controlled  our own `web_search` tool on Tavily with: a domain blocklist (contamination),
              a per-sample query budget, and a disk cache keyed by query.
  replay      `controlled`, but answered only from the disk cache: no network, so runs are
              repeatable and every model sees identical results.

All queries and results are in the Inspect log as tool calls; `controlled`/`replay` also count
queries in the sample store (key "search_queries").
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import httpx
from inspect_ai.tool import Tool, ToolError, tool, web_search
from inspect_ai.util import store

SEARCH_MODES = ("off", "native", "tavily", "controlled", "replay")

# Benchmark mirrors and answer leaks. Extend per benchmark with blocklist=[...].
DEFAULT_BLOCKLIST = [
    "huggingface.co",
    "hf.co",
    "paperswithcode.com",
    "kaggle.com",
]

DEFAULT_CACHE = Path(os.environ.get("LABKIT_SEARCH_CACHE", Path(__file__).resolve().parents[3] / "data" / "search_cache"))


def _blocked(url: str, blocklist: list[str]) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in blocklist)


def _cache_path(cache_dir: Path, query: str, max_results: int) -> Path:
    key = hashlib.sha256(json.dumps([query.strip().lower(), max_results]).encode()).hexdigest()[:24]
    return cache_dir / f"{key}.json"


async def _tavily(query: str, max_results: int, exclude: list[str]) -> list[dict]:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise ToolError("Search backend is not configured (TAVILY_API_KEY missing).")
    async with httpx.AsyncClient(timeout=30) as client:  # honours the cluster's https_proxy
        r = await client.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"query": query, "max_results": max_results, "exclude_domains": exclude,
                  "search_depth": "basic", "include_answer": False},
        )
    if r.status_code != 200:
        raise ToolError(f"Search failed (HTTP {r.status_code}).")
    return [{"title": x.get("title", ""), "url": x.get("url", ""), "content": x.get("content", "")}
            for x in r.json().get("results", [])]


@tool(name="web_search")
def controlled_search(
    blocklist: list[str] | None = None,
    max_queries: int = 10,
    max_results: int = 5,
    cache_dir: str | Path | None = None,
    replay: bool = False,
) -> Tool:
    blocklist = [d.lower() for d in (DEFAULT_BLOCKLIST if blocklist is None else blocklist)]
    cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE

    async def execute(query: str) -> str:
        """Search the web and return the top results (title, URL, snippet).

        Args:
            query: The search query.
        """
        used = store().get("search_queries", 0)
        if used >= max_queries:
            return f"Search budget exhausted ({max_queries} queries per task)."
        store().set("search_queries", used + 1)

        path = _cache_path(cache, query, max_results)
        if path.exists():
            results = json.loads(path.read_text())["results"]
        elif replay:
            results = []
        else:
            results = await _tavily(query, max_results, blocklist)
            cache.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"query": query, "results": results}, indent=1))

        # Filter again at read time so a cache made with a looser blocklist cannot leak.
        results = [x for x in results if not _blocked(x["url"], blocklist)]
        if not results:
            return "No results."
        return "\n\n".join(f"[{i}] {x['title']}\n{x['url']}\n{x['content']}" for i, x in enumerate(results, 1))

    return execute


def search_tools(mode: str = "off", **kwargs) -> list[Tool]:
    """Search tool list for a mode in SEARCH_MODES (kwargs go to controlled_search)."""
    if mode in ("off", False, None):  # YAML parses a bare `off` as False
        return []
    if mode == "native":
        return [web_search({"openai": True, "anthropic": True, "gemini": True, "grok": True, "perplexity": True})]
    if mode == "tavily":
        return [web_search({"tavily": True})]
    if mode == "controlled":
        return [controlled_search(**kwargs)]
    if mode == "replay":
        return [controlled_search(replay=True, **kwargs)]
    raise ValueError(f"search mode must be one of {SEARCH_MODES}, got {mode!r}")
