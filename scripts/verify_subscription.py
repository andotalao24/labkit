"""Re-audit subscription runs (tasks/extraction_sub.py) for any web search / internet access.

  .venv/bin/python scripts/verify_subscription.py logs/extraction_sub/<file>.eval   # every sample in a log
  .venv/bin/python scripts/verify_subscription.py logs/subscription/runs/<run_dir>  # one saved run

For each run directory (events.jsonl, egress.jsonl, codex_sessions/) it checks:
  claude: the init event's tool list is within the allowed list and has no search/fetch tool, no
          MCP servers, no tool_use / server_tool_use of a search tool, server_tool_use counters 0.
  codex:  no web_search event items, no web_search_call (or other search-like) response item in the
          rollout files.
  both:   every connection that left the sandbox went to an allowlisted model/login host (denied
          attempts are listed: they were blocked).
Exits 1 if anything is found.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from labkit.subscription import audit_run


def runs_from(arg: str) -> list[tuple[str, Path]]:
    p = Path(arg)
    if p.is_dir():
        audit = json.loads((p / "audit.json").read_text()) if (p / "audit.json").exists() else {}
        agent = audit.get("agent") or ("codex" if "__codex" in p.name else "claude")
        return [(agent, p)]
    from inspect_ai.log import read_eval_log

    log = read_eval_log(arg)
    out = []
    for s in log.samples or []:
        md = s.metadata or {}
        if not md.get("run_dir"):
            print(f"{arg} {s.id}: no run_dir in metadata (sample errored before the CLI ran?)")
            continue
        out.append((md["harness"].split("-")[0], Path(md["run_dir"])))
    return out


def main(args: list[str]) -> int:
    rc = 0
    for a in args:
        for agent, run in runs_from(a):
            r = audit_run(agent, run)
            print(f"{run}\n  agent={agent} model={r.get('model')}")
            if agent == "claude":
                print(f"  tools offered: {r.get('tools_offered')}  used: {r.get('tools_used')}")
            else:
                print(f"  tools offered: {r.get('tools_offered')}")
                print(f"  items/tools used: {r.get('items_and_tools')}")
            print(f"  egress allowed: {r['egress_allowed']}  denied: {r['egress_denied']}")
            if r["ok"]:
                print("  OK: no search tool offered or used; no egress beyond the model/login hosts")
            else:
                rc = 1
                print("  FAIL:\n   - " + "\n   - ".join(r["problems"]))
    return rc


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1:]))
