"""Prove from an Inspect log that the model could not search the web.

  .venv/bin/python scripts/verify_no_tools.py logs/extraction/<file>.eval          # plain: no tools at all
  .venv/bin/python scripts/verify_no_tools.py --agent logs/extraction_codex/<f>.eval  # agent: tools ok, no search

Checks every model call's raw provider request and response.
  plain mode:  the request has no tools / functions / tool_choice; no tool calls anywhere.
  agent mode:  tools are allowed (shell, apply_patch, ...) but none may be search/web-like, the raw
               request's tool list has no hosted web_search tool, and nothing in the transcript
               called one.
  both:        response output items are only reasoning/message/function calls (no web_search_call),
               and OpenAI's usage counter tool_usage.web_search.num_requests is 0.
Exits 1 if anything is found.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

from inspect_ai.log import read_eval_log

SEARCH_LIKE = re.compile(r"search|web|brows|fetch|http|url|internet", re.I)
OK_OUTPUT_TYPES = {"reasoning", "message", "function_call", "custom_tool_call", "local_shell_call"}


def tool_names(tools) -> list[str]:
    names = []
    for t in tools or []:
        if isinstance(t, dict):
            names.append(t.get("name") or (t.get("function") or {}).get("name") or t.get("type") or "?")
        else:
            names.append(getattr(t, "name", "?"))
    return names


def check(path: str, agent: bool) -> int:
    log = read_eval_log(path)
    problems: list[str] = []
    calls, offered = 0, set()
    for s in log.samples or []:
        for ev in s.events:
            if ev.event != "model":
                continue
            # audit the tested model only (a judge's calls are appended to the log when scoring)
            if ev.model.split("/")[-1] != log.eval.model.split("/")[-1]:
                continue
            calls += 1
            names = [t.name for t in ev.tools or []]
            offered.update(names)
            if names and not agent:
                problems.append(f"{s.id}: tools offered {names}")
            if agent and (bad := [n for n in names if SEARCH_LIKE.search(n)]):
                problems.append(f"{s.id}: search-like tools offered {bad}")
            if ev.call is None:
                problems.append(f"{s.id}: no raw API call recorded (cannot verify request)")
                continue
            req = ev.call.request or {}
            if not agent:
                for key in ("tools", "functions", "tool_choice"):
                    if req.get(key):
                        problems.append(f"{s.id}: raw request has {key}={json.dumps(req[key])[:200]}")
            for t in req.get("tools") or []:
                if isinstance(t, dict) and (SEARCH_LIKE.search(str(t.get("type", ""))) or
                                            SEARCH_LIKE.search(tool_names([t])[0])):
                    problems.append(f"{s.id}: raw request offers {json.dumps(t)[:160]}")
            resp = ev.call.response or {}
            for item in resp.get("output", []) or []:
                if item.get("type") not in OK_OUTPUT_TYPES or (not agent and item.get("type") != "reasoning"
                                                                and item.get("type") != "message"):
                    problems.append(f"{s.id}: response output item of type {item.get('type')!r}")
            for name, u in (resp.get("tool_usage") or {}).items():
                n = (u or {}).get("num_requests", 0) if isinstance(u, dict) else u
                if n:
                    problems.append(f"{s.id}: provider reports {name} used {n} time(s)")
            body = json.dumps(resp)
            if '"type": "server_tool_use"' in body or '"type": "web_search' in body:
                problems.append(f"{s.id}: response contains a server-side search block")
            for ch in resp.get("choices", []) or []:
                if not agent and (ch.get("message") or {}).get("tool_calls"):
                    problems.append(f"{s.id}: response choice has tool_calls")
        for msg in s.messages:
            for tc in getattr(msg, "tool_calls", None) or []:
                if not agent or SEARCH_LIKE.search(tc.function):
                    problems.append(f"{s.id}: transcript tool call {tc.function}")
            if not agent and msg.role == "tool":
                problems.append(f"{s.id}: transcript contains tool results")
    mode = "agent" if agent else "plain"
    print(f"{path}\n  mode={mode} samples={len(log.samples or [])} model_calls={calls} model={log.eval.model}")
    if agent:
        print(f"  tools offered to the model: {sorted(offered)}")
    if problems:
        print("  FAIL:\n   - " + "\n   - ".join(dict.fromkeys(problems)))
        return 1
    print("  OK: " + ("no search tool offered or used; " if agent else "no tools offered, no tool calls; ")
          + "provider web_search counter 0 in every response")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", action="store_true", help="tools allowed, but no search")
    ap.add_argument("logs", nargs="+")
    a = ap.parse_args()
    sys.exit(max(check(p, a.agent) for p in a.logs))
