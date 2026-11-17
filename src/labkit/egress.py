"""Allowlisted egress for sandboxes that are otherwise offline.

The Apptainer instance keeps `--net --network none` (loopback only). Its single way out is a Unix
socket bind-mounted into the container:

  CLI in container --HTTPS_PROXY--> 127.0.0.1:<port> (FORWARDER_PY, runs inside the container)
      --unix socket--> EgressProxy (host, this module) --CONNECT--> allowlisted host:443 only

EgressProxy only accepts `CONNECT host:443` for hosts on the allowlist; everything else (other
hosts, other ports, plain-HTTP requests) gets 403 and is logged. Every attempt, allowed or not, is
appended to a JSONL log, so a run shows exactly which hosts were contacted. If the host itself sits
behind a proxy (https_proxy), allowed connections are chained through it.

Used by labkit.subscription to let the Codex / Claude Code CLIs reach their own model backend with
a subscription login while bash/python inside the sandbox still cannot reach anything else.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

# hosts each CLI needs for subscription (OAuth) inference + token refresh; nothing else
ALLOW = {
    "codex": ("chatgpt.com", "auth.openai.com"),
    "claude": ("api.anthropic.com", "platform.claude.com", "console.anthropic.com"),
}

# Runs inside the container (stdlib only): 127.0.0.1:PORT <-> unix socket. argv: sock port ready_file
FORWARDER_PY = r'''
import asyncio, sys
SOCK, PORT, READY = sys.argv[1], int(sys.argv[2]), sys.argv[3]
async def pipe(r, w):
    try:
        while (b := await r.read(65536)):
            w.write(b); await w.drain()
    except Exception:
        pass
    finally:
        try: w.close()
        except Exception: pass
async def handle(cr, cw):
    try:
        ur, uw = await asyncio.open_unix_connection(SOCK)
    except Exception:
        cw.close(); return
    await asyncio.gather(pipe(cr, uw), pipe(ur, cw))
async def main():
    srv = await asyncio.start_server(handle, "127.0.0.1", PORT)
    open(READY, "w").close()
    async with srv:
        await srv.serve_forever()
asyncio.run(main())
'''


def _upstream_proxy() -> tuple[str, int] | None:
    raw = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
    if not raw:
        return None
    u = urlparse(raw if "://" in raw else f"http://{raw}")
    return (u.hostname, u.port or 3128) if u.hostname else None


def host_allowed(host: str, allow: tuple[str, ...]) -> bool:
    host = host.lower().rstrip(".")
    return any(host == a or (a.startswith("*.") and host.endswith(a[1:])) for a in allow)


class EgressProxy:
    def __init__(self, sock_path: str, allow: tuple[str, ...], log_path: str | Path) -> None:
        self.sock_path, self.allow, self.log_path = sock_path, tuple(allow), Path(log_path)
        self.upstream = _upstream_proxy()
        self._server: asyncio.base_events.Server | None = None
        self.events: list[dict] = []

    async def __aenter__(self) -> "EgressProxy":
        if os.path.exists(self.sock_path):
            os.unlink(self.sock_path)
        self._server = await asyncio.start_unix_server(self._handle, path=self.sock_path)
        os.chmod(self.sock_path, 0o600)  # only our user (the container runs as us)
        return self

    async def __aexit__(self, *exc) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        if os.path.exists(self.sock_path):
            os.unlink(self.sock_path)

    def _log(self, ev: dict) -> None:
        ev = {"ts": round(time.time(), 3), **ev}
        self.events.append(ev)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(ev) + "\n")

    async def _handle(self, cr: asyncio.StreamReader, cw: asyncio.StreamWriter) -> None:
        target = "?"
        try:
            head = await asyncio.wait_for(cr.readuntil(b"\r\n\r\n"), 30)
            parts = head.split(b"\r\n", 1)[0].decode("latin-1").split()
            method, target = (parts + ["", ""])[:2]
            host, _, port = target.rpartition(":")
            ok = method == "CONNECT" and port == "443" and host_allowed(host, self.allow)
            if not ok:
                self._log({"method": method, "target": target, "allowed": False})
                cw.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
                await cw.drain()
                return
            if self.upstream:
                ur, uw = await asyncio.wait_for(asyncio.open_connection(*self.upstream), 30)
                uw.write(f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode())
                await uw.drain()
                resp = await asyncio.wait_for(ur.readuntil(b"\r\n\r\n"), 30)
                if b" 200" not in resp.split(b"\r\n", 1)[0]:
                    raise ConnectionError(resp.split(b"\r\n", 1)[0].decode("latin-1"))
            else:
                ur, uw = await asyncio.wait_for(asyncio.open_connection(host, 443), 30)
            cw.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
            await cw.drain()
            up, down = await asyncio.gather(self._pipe(cr, uw), self._pipe(ur, cw))
            self._log({"method": method, "target": target, "allowed": True, "bytes_up": up, "bytes_down": down})
        except Exception as e:  # noqa: BLE001
            self._log({"target": target, "allowed": None, "error": f"{type(e).__name__}: {e}"[:300]})
        finally:
            try:
                cw.close()
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    async def _pipe(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> int:
        n = 0
        try:
            while b := await r.read(65536):
                n += len(b)
                w.write(b)
                await w.drain()
        except Exception:  # noqa: BLE001
            pass
        finally:
            try:
                w.close()
            except Exception:  # noqa: BLE001
                pass
        return n
