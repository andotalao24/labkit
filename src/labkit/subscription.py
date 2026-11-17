"""Run Codex CLI or Claude Code on a *subscription* login (ChatGPT / Claude OAuth), not API keys.

Inspect's usual agent bridge (inspect_swe) routes the CLI's model calls through an Inspect model,
i.e. an API key. Here the CLI talks to its own backend with the host's existing login instead:

  host ~/.codex/auth.json            (codex login, ChatGPT plan)   -> copied into the sandbox home
  host ~/.claude/.credentials.json   (claude /login, Claude plan)  -> copied into the sandbox home

Each sample runs the CLI once, headless, inside the labkit Apptainer sandbox:
  * --contain --cleanenv --no-mount bind-paths,cwd: no host env, $HOME, /projects (ground truth
    lives there and never enters the sandbox; only task.md + images are copied to /workspace).
  * --net --network none, plus ONE way out: labkit.egress, an allowlist proxy that only accepts
    CONNECT to the CLI's own backend (ALLOW in egress.py). curl/pip/python from the agent's shell
    to anything else get 403, and every attempt is logged.
  * web search is switched off in the CLI itself (it is a server-side tool, so the network layer
    alone cannot block it): Codex `-c web_search="disabled"`; Claude Code an explicit `--tools`
    list without WebSearch/WebFetch, `--disallowedTools WebSearch,WebFetch`, no MCP servers.
  * audit_run() then checks the CLI's own event stream (and Codex's rollout files) for any search
    tool offered or used; the sample fails if anything is found. scripts/verify_subscription.py
    re-runs the same audit on saved runs.

Saved per sample under logs/subscription/runs/<sample>__<agent>__<time>/: events.jsonl (CLI JSON
stream), stderr.txt, egress.jsonl, prompt, codex rollout files, audit.json, answer.md. Credentials
are never copied there. If the CLI refreshed its OAuth token in the sandbox, the newer token file is
written back to the host so the host login keeps working (refresh tokens rotate).
"""
from __future__ import annotations

import atexit
import base64
import fcntl
import json
import os
import re
import shlex
import shutil
import tempfile
import time
from pathlib import Path
from typing import Literal

from inspect_ai.model import ChatMessageAssistant, ContentImage, ContentText, ModelOutput
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox

from labkit.egress import ALLOW, FORWARDER_PY, EgressProxy
from labkit.sandbox.apptainer import ApptainerConfig, ApptainerSandboxEnvironment

Agent = Literal["codex", "claude"]
ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "logs" / "subscription" / "runs"
BIN_IN = "/opt/agents/bin"
EGRESS_IN = "/opt/egress"
PORT = 3128
SEARCH_LIKE = re.compile(r"search|web|brows|fetch|http|url|internet", re.I)
CLAUDE_TOOLS = ("Bash", "Read", "Write", "Edit", "Glob", "Grep", "TodoWrite")
CLAUDE_BLOCKED = ("WebSearch", "WebFetch")

# one host dir per Inspect process holding the egress sockets; bound into every sandbox
_EGRESS_DIR = tempfile.mkdtemp(prefix="labkit-egress-",
                               dir=os.environ.get("LABKIT_SANDBOX_TMP") or "/tmp")
atexit.register(shutil.rmtree, _EGRESS_DIR, ignore_errors=True)


# ---- host binaries and credentials ----------------------------------------------------------------
def _binaries(agent: Agent) -> dict[str, str]:
    """{name in container: host path}. The CLIs are bound read-only from the host install."""
    if agent == "claude":
        p = os.environ.get("LABKIT_CLAUDE_BIN") or shutil.which("claude")
        if not p:
            raise FileNotFoundError("claude CLI not found (set LABKIT_CLAUDE_BIN)")
        return {"claude": os.path.realpath(p)}
    p = os.environ.get("LABKIT_CODEX_BIN")
    if not p:  # npm package: bin/codex.js wraps a static musl binary; bind the binary itself
        js = shutil.which("codex")
        if not js:
            raise FileNotFoundError("codex CLI not found (set LABKIT_CODEX_BIN)")
        pkg = Path(os.path.realpath(js)).parents[1]
        hits = sorted(pkg.glob("node_modules/@openai/codex-linux-x64/vendor/*/bin/codex")) or \
            sorted(pkg.glob("vendor/*/bin/codex")) or sorted(pkg.glob("vendor/*/codex/codex"))
        if not hits:
            raise FileNotFoundError(f"no native codex binary under {pkg} (set LABKIT_CODEX_BIN)")
        p = str(hits[0])
    p = os.path.realpath(p)
    out = {"codex": p}
    for sib in Path(p).parent.iterdir():  # helpers Codex spawns from its own dir (codex-code-mode-host)
        if sib.is_file() and os.access(sib, os.X_OK):
            out[sib.name] = str(sib)
    rg = Path(p).parents[1] / "codex-path" / "rg"
    if rg.exists():
        out["rg"] = str(rg)
    return out


def _host_creds(agent: Agent) -> Path:
    if agent == "codex":
        p = Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser() / "auth.json"
        hint = "run `codex login` (ChatGPT sign-in) on this machine"
    else:
        p = Path(os.environ.get("CLAUDE_CONFIG_DIR", "~/.claude")).expanduser() / ".credentials.json"
        hint = "run `claude` and /login with your Claude subscription on this machine"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: {hint}")
    d = json.loads(p.read_text())
    if agent == "codex" and (d.get("auth_mode") != "chatgpt" or d.get("OPENAI_API_KEY")):
        raise RuntimeError(f"{p} is not a ChatGPT subscription login (auth_mode={d.get('auth_mode')})")
    if agent == "claude" and not (d.get("claudeAiOauth") or {}).get("refreshToken"):
        raise RuntimeError(f"{p} has no Claude subscription OAuth login")
    return p


def _staged_creds(agent: Agent, home: Path) -> Path:
    return home / (".codex/auth.json" if agent == "codex" else ".claude/.credentials.json")


def _sync_creds_back(agent: Agent, staged: Path, host: Path) -> bool:
    """Write a token the CLI refreshed inside the sandbox back to the host (only if newer)."""
    try:
        new = json.loads(staged.read_text())
    except Exception:  # noqa: BLE001
        return False
    with open(str(host) + ".labkit.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        old = json.loads(host.read_text())
        if new == old:
            return False
        if agent == "claude":
            newer = (new.get("claudeAiOauth") or {}).get("expiresAt", 0) > \
                (old.get("claudeAiOauth") or {}).get("expiresAt", 0)
        else:
            newer = str(new.get("last_refresh") or "") > str(old.get("last_refresh") or "")
        if newer:
            tmp = host.with_name(host.name + ".labkit.tmp")
            tmp.write_text(json.dumps(new, indent=2))
            os.chmod(tmp, 0o600)
            os.replace(tmp, host)
        return newer


# ---- sandbox config ---------------------------------------------------------------------------
def subscription_sandbox(agent: Agent, image: str | None = None) -> tuple[str, ApptainerConfig]:
    binds = [f"{_EGRESS_DIR}:{EGRESS_IN}"]
    binds += [f"{host}:{BIN_IN}/{name}:ro" for name, host in _binaries(agent).items()]
    kw = {"image": image} if image else {}
    return "apptainer", ApptainerConfig(network=False, binds=tuple(binds), **kw)


# ---- prompt ------------------------------------------------------------------------------------
def _image_bytes(img: ContentImage) -> bytes:
    src = img.image
    if src.startswith("data:"):
        return base64.b64decode(src.split(",", 1)[1])
    return Path(src).read_bytes()


def _parts(state: TaskState) -> list[tuple[str, str | bytes]]:
    """User input as an ordered list of ("text", str) / ("image", png bytes)."""
    out: list[tuple[str, str | bytes]] = []
    for m in state.messages:
        if m.role != "user":
            continue
        for c in [ContentText(text=m.content)] if isinstance(m.content, str) else m.content:
            if isinstance(c, ContentText):
                out.append(("text", c.text))
            elif isinstance(c, ContentImage):
                out.append(("image", _image_bytes(c)))
            else:
                raise ValueError(f"unsupported input content {type(c).__name__}")
    return out


def _cli_command(agent: Agent, model: str | None, effort: str | None, images: list[str],
                 run: str) -> tuple[str, str]:
    """(shell command, stdin file name in run dir)."""
    q = shlex.quote
    if agent == "codex":
        cmd = ["codex", "exec", "--json", "--color", "never", "--skip-git-repo-check",
               "--dangerously-bypass-approvals-and-sandbox",  # the Apptainer instance is the sandbox
               "-C", "/workspace", "-o", f"{run}/last_message.txt",
               "-c", 'web_search="disabled"',
               "-c", 'shell_environment_policy.exclude=["*PROXY*","*proxy*"]']
        if model:
            cmd += ["-m", model]
        if effort:
            cmd += ["-c", f'model_reasoning_effort="{effort}"']
        for img in images:  # repeated flag; prompt comes from stdin so nothing follows to be swallowed
            cmd += ["--image", img]
        return " ".join(q(c) for c in cmd), "prompt.txt"
    cmd = ["claude", "-p", "--input-format", "stream-json", "--output-format", "stream-json",
           "--verbose", "--tools", ",".join(CLAUDE_TOOLS), "--disallowedTools", ",".join(CLAUDE_BLOCKED),
           "--strict-mcp-config", "--dangerously-skip-permissions", "--no-session-persistence"]
    if model:
        cmd += ["--model", model]
    if effort:
        cmd += ["--effort", effort]
    return " ".join(q(c) for c in cmd), "prompt.jsonl"


# ---- audit --------------------------------------------------------------------------------------
def _jsonl(p: Path) -> list[dict]:
    out = []
    if p.exists():
        for line in p.read_text(errors="replace").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _codex_tool_names(tools) -> list[str]:
    """Flatten Codex's additional_tools list into "type:name" (namespaces expanded)."""
    out = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        out.append(f"{t.get('type')}:{t.get('name', '')}")
        for k in ("tools", "functions"):
            out += _codex_tool_names(t.get(k))
    return out


def audit_run(agent: Agent, run: Path) -> dict:
    """No-search audit of one saved run directory. problems == [] means clean."""
    problems: list[str] = []
    info: dict = {"agent": agent}
    events = _jsonl(run / "events.jsonl")
    if not events:
        problems.append("no CLI events recorded (cannot verify)")
    if agent == "claude":
        init = next((e for e in events if e.get("type") == "system" and e.get("subtype") == "init"), None)
        if init is None:
            problems.append("no init event (cannot see which tools were offered)")
        else:
            tools = init.get("tools") or []
            info.update(tools_offered=tools, model=init.get("model"), mcp_servers=init.get("mcp_servers"))
            if extra := [t for t in tools if t not in CLAUDE_TOOLS]:
                problems.append(f"tools offered beyond the allowed list: {extra}")
            if bad := [t for t in tools if SEARCH_LIKE.search(t)]:
                problems.append(f"search-like tools offered: {bad}")
            if init.get("mcp_servers"):
                problems.append(f"MCP servers attached: {init['mcp_servers']}")
        used = []
        for e in events:
            for b in ((e.get("message") or {}).get("content") or []) if e.get("type") == "assistant" else []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_use":
                    used.append(b.get("name"))
                    if SEARCH_LIKE.search(b.get("name") or ""):
                        problems.append(f"search-like tool called: {b.get('name')}")
                if b.get("type") in ("server_tool_use", "web_search_tool_result", "web_fetch_tool_result"):
                    problems.append(f"server-side tool block in response: {b.get('type')}")
            if e.get("type") == "result":
                stu = ((e.get("usage") or {}).get("server_tool_use") or {})
                for k, v in stu.items():
                    if v:
                        problems.append(f"provider reports {k}={v}")
        info["tools_used"] = sorted(set(u for u in used if u))
    else:
        rollouts = sorted((run / "codex_sessions").rglob("*.jsonl"))
        info["rollout_files"] = [str(r.relative_to(run)) for r in rollouts]
        if not rollouts:
            problems.append("no Codex rollout file saved (cannot verify response items)")
        used, offered = [], []
        for e in events:
            item = e.get("item") or {}
            if item.get("type") and SEARCH_LIKE.search(item["type"]):
                problems.append(f"Codex event item of type {item['type']}")
            if item.get("type"):
                used.append(item["type"])
        for r in rollouts:
            for e in _jsonl(r):
                p = e.get("payload") or {}
                t = p.get("type") or ""
                if e.get("type") == "turn_context" and p.get("model"):
                    info["model"] = p["model"]
                if e.get("type") == "response_item" and t == "additional_tools":
                    offered += _codex_tool_names(p.get("tools"))
                if e.get("type") == "response_item":
                    if t in ("web_search_call",) or (t.endswith("_call") and SEARCH_LIKE.search(t)):
                        problems.append(f"rollout response item {t}")
                    if t == "function_call" and SEARCH_LIKE.search(p.get("name") or ""):
                        problems.append(f"search-like function call {p.get('name')}")
                    if t in ("function_call", "custom_tool_call", "local_shell_call"):
                        used.append(p.get("name") or t)
            if '"web_search_call"' in r.read_text(errors="replace"):
                problems.append(f"{r.name}: contains a web_search_call")
        info["items_and_tools"] = sorted(set(used))
        info["tools_offered"] = sorted(set(offered))
        if not offered:
            problems.append("no tool list (additional_tools) in the rollout (cannot see what was offered)")
        if bad := [t for t in set(offered) if SEARCH_LIKE.search(t)]:
            problems.append(f"search-like tools offered: {sorted(bad)}")
    egress = _jsonl(run / "egress.jsonl")
    allow = ALLOW[agent]
    info["egress_allowed"] = sorted({e["target"] for e in egress if e.get("allowed")})
    info["egress_denied"] = sorted({str(e.get("target")) for e in egress if e.get("allowed") is False})
    for t in info["egress_allowed"]:
        if t.rpartition(":")[0] not in allow:
            problems.append(f"egress to non-allowlisted host {t}")
    info["problems"] = list(dict.fromkeys(problems))
    info["ok"] = not info["problems"]
    return info


def _answer(agent: Agent, run: Path) -> tuple[str, dict]:
    usage: dict = {}
    events = _jsonl(run / "events.jsonl")
    if agent == "codex":
        for e in events:
            if e.get("type") == "turn.completed":
                usage = e.get("usage") or usage
        p = run / "last_message.txt"
        text = p.read_text() if p.exists() else ""
        if not text:  # fall back to the last agent message in the stream
            msgs = [(e.get("item") or {}).get("text") for e in events
                    if (e.get("item") or {}).get("type") == "agent_message"]
            text = next((m for m in reversed(msgs) if m), "")
        return text, usage
    res = next((e for e in reversed(events) if e.get("type") == "result"), {})
    usage = {k: res.get(k) for k in ("usage", "total_cost_usd", "num_turns", "duration_ms", "modelUsage",
                                     "is_error", "subtype") if k in res}
    return res.get("result") or "", usage


# ---- solver -------------------------------------------------------------------------------------
@solver
def subscription_agent(agent: Agent = "codex", model: str | None = None, effort: str | None = "high",
                       time_limit: int = 3600) -> Solver:
    """Run `agent` (codex | claude) once on the sample's user input, with the host's subscription
    login. Needs the sandbox from subscription_sandbox(agent). model/effort: None = CLI default."""
    if agent not in ("codex", "claude"):
        raise ValueError(f"agent must be codex or claude, not {agent!r}")
    bin_names = list(_binaries(agent))

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        try:
            sb = sandbox().as_type(ApptainerSandboxEnvironment)
        except TypeError as e:
            raise RuntimeError("subscription_agent needs the apptainer sandbox (subscription_sandbox())") from e
        host_creds = _host_creds(agent)
        sbx_home = Path(sb.workdir) / "home"
        sbx_ws = Path(sb.workdir) / "ws"
        home_in = os.path.expanduser("~")  # the sandbox mounts its own empty home at the same path
        run_in = f"{home_in}/.labkit_run"
        run_host = sbx_home / ".labkit_run"
        run_host.mkdir(parents=True)

        # credentials -> sandbox home (not /workspace); 0600
        staged = _staged_creds(agent, sbx_home)
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(host_creds, staged)
        os.chmod(staged, 0o600)

        # task files -> /workspace; prompt -> run dir
        parts = _parts(state)
        names = list(state.metadata.get("images") or [])
        images: list[str] = []
        blocks: list[dict] = []
        texts: list[str] = []
        for kind, val in parts:
            if kind == "text":
                texts.append(val)  # type: ignore[arg-type]
                blocks.append({"type": "text", "text": val})
            else:
                name = names[len(images)] if len(images) < len(names) else f"image{len(images) + 1}.png"
                (sbx_ws / name).write_bytes(val)  # type: ignore[arg-type]
                images.append(f"/workspace/{name}")
                blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                           "data": base64.b64encode(val).decode()}})  # type: ignore[arg-type]
        if agent == "codex":
            # Codex attaches all images after the text; label the order the way plain runs see it
            prompt = "\n\n".join(t for t in texts if not re.fullmatch(r"\s*image\d+\.\w+:\s*", t))
            if images:
                prompt += "\n\nAttached images, in order: " + ", ".join(Path(i).name for i in images) + "."
            (run_host / "prompt.txt").write_text(prompt)
        else:
            (run_host / "prompt.jsonl").write_text(json.dumps(
                {"type": "user", "message": {"role": "user", "content": blocks}}) + "\n")
        (run_host / "fwd.py").write_text(FORWARDER_PY)

        cli, stdin_name = _cli_command(agent, model, effort, images, run_in)
        sid = os.urandom(6).hex()
        sock_host = os.path.join(_EGRESS_DIR, f"{sid}.sock")
        proxy_env = f"http://127.0.0.1:{PORT}"
        script = f"""set -u
export PATH={BIN_IN}:$PATH HOME={home_in}
export CODEX_HOME={home_in}/.codex
export HTTPS_PROXY={proxy_env} https_proxy={proxy_env} HTTP_PROXY={proxy_env} http_proxy={proxy_env}
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_AUTOUPDATER=1 DISABLE_TELEMETRY=1
export ENABLE_CLAUDEAI_MCP_SERVERS=false IS_SANDBOX=1
python3 {run_in}/fwd.py {EGRESS_IN}/{sid}.sock {PORT} {run_in}/.fwd_ready 2>{run_in}/fwd.err &
FWD=$!
for i in $(seq 200); do [ -e {run_in}/.fwd_ready ] && break; sleep 0.05; done
[ -e {run_in}/.fwd_ready ] || {{ echo "egress forwarder did not start" >&2; exit 97; }}
cd /workspace
timeout -k 30 {int(time_limit)} {cli} < {run_in}/{stdin_name} > {run_in}/events.jsonl 2> {run_in}/stderr.txt
rc=$?
echo $rc > {run_in}/exit_code
kill $FWD 2>/dev/null
exit $rc
"""
        (run_host / "run.sh").write_text(script)

        stamp = time.strftime("%Y%m%d-%H%M%S")
        out = RUNS / f"{state.sample_id}__{agent}{'_' + model if model else ''}__{stamp}"
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            async with EgressProxy(sock_host, ALLOW[agent], out / "egress.jsonl"):
                r = await sb.exec(["bash", f"{run_in}/run.sh"], timeout=time_limit + 120)
        finally:
            refreshed = _sync_creds_back(agent, staged, host_creds)
            for f in ("events.jsonl", "stderr.txt", "exit_code", "last_message.txt", "fwd.err", "run.sh",
                      stdin_name):
                if (run_host / f).exists():
                    shutil.copyfile(run_host / f, out / f)
            if agent == "codex" and (sbx_home / ".codex" / "sessions").exists():
                shutil.copytree(sbx_home / ".codex" / "sessions", out / "codex_sessions", dirs_exist_ok=True)

        answer, usage = _answer(agent, out)
        audit = audit_run(agent, out)
        (out / "answer.md").write_text(answer)
        (out / "audit.json").write_text(json.dumps(audit, indent=1))
        actual_model = f"{agent}-sub/{audit.get('model') or model or 'default'}"
        state.metadata.update({
            "harness": f"{agent}-subscription", "answer_model": actual_model, "run_dir": str(out),
            "exit_code": r.returncode, "elapsed_s": round(time.time() - t0, 1), "usage": usage,
            "no_search_audit": audit, "token_refreshed_back_to_host": refreshed,
            "binaries": bin_names,
        })
        if not audit["ok"]:
            raise RuntimeError(f"no-search audit failed ({out}): {audit['problems']}")
        if r.returncode != 0 or not answer.strip():
            tail = (out / "stderr.txt").read_text(errors="replace")[-2000:] if (out / "stderr.txt").exists() else ""
            raise RuntimeError(f"{agent} exited {r.returncode} (answer {len(answer)} chars), see {out}:\n{tail}"
                               f"\n{r.stderr[-1000:]}")
        state.messages.append(ChatMessageAssistant(content=answer, model=actual_model))
        state.output = ModelOutput.from_content(model=actual_model, content=answer)
        return state

    return solve
