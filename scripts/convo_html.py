"""Render an Inspect conversation (e.g. a follow-up chain) as one readable HTML page.

  .venv/bin/python scripts/convo_html.py logs/followup/<last>.eval -o logs/followup/T2_conversation.html \
      [--context-turns 2] [--title "..."]

The first --context-turns messages (the original task + answer) are shown collapsed; the rest are
shown as the conversation. Markdown and LaTeX (\\( \\), \\[ \\]) render in the browser (marked + KaTeX).
Images are embedded, so the file is self-contained apart from the two CDN libraries.
Provider-encrypted reasoning is not readable and is noted, not shown.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from inspect_ai.log import read_eval_log


def parts(msg) -> tuple[list[tuple[str, str]], int]:
    """[(kind, payload)] for text/image parts, and the count of hidden reasoning blocks."""
    out, hidden = [], 0
    content = msg.content if isinstance(msg.content, list) else [msg.content]
    for c in content:
        if isinstance(c, str):
            out.append(("md", c))
        elif c.type == "text":
            out.append(("md", c.text))
        elif c.type == "image":
            out.append(("img", c.image))
        elif c.type == "reasoning":
            hidden += 1
    return out, hidden


def render_turn(msg, model: str, label: str | None = None) -> str:
    ps, hidden = parts(msg)
    role = "user" if msg.role == "user" else "assistant"
    who = label or ("Question" if role == "user" else model)
    body = []
    pending_caption = None
    for kind, payload in ps:
        if kind == "md":
            if payload.strip().endswith(".png:") and len(payload) < 40:  # image caption line
                pending_caption = payload.strip().rstrip(":")
                continue
            body.append(f'<div class="md">{html.escape(payload)}</div>')
        else:
            cap = f"<figcaption>{html.escape(pending_caption)}</figcaption>" if pending_caption else ""
            body.append(f'<figure><img src="{payload}" alt="{html.escape(pending_caption or "image")}">{cap}</figure>')
            pending_caption = None
    imgs = [b for b in body if b.startswith("<figure")]
    texts = [b for b in body if not b.startswith("<figure")]
    gallery = f'<div class="gallery">{"".join(imgs)}</div>' if imgs else ""
    note = (f'<p class="note">{hidden} reasoning block(s) hidden: encrypted by the provider, '
            f'passed back to the model but not readable.</p>') if hidden else ""
    return (f'<section class="turn {role}"><div class="who">{html.escape(who)}</div>'
            f'{"".join(texts)}{gallery}{note}</section>')


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
<style>
:root{--bg:#f7f6f3;--card:#ffffff;--ink:#1f2328;--muted:#6a6f76;--line:#e3e1dc;--user:#eef3fb;--userline:#c9d8ef;--accent:#2f5d9e;--code:#f2f1ee}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#16181b;--card:#1e2125;--ink:#e6e6e3;--muted:#9aa0a6;--line:#2e3238;--user:#1d2633;--userline:#2f4566;--accent:#8db4ec;--code:#262a2f}}
:root[data-theme="dark"]{--bg:#16181b;--card:#1e2125;--ink:#e6e6e3;--muted:#9aa0a6;--line:#2e3238;--user:#1d2633;--userline:#2f4566;--accent:#8db4ec;--code:#262a2f}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
main{max-width:860px;margin:0 auto;padding:32px 16px 80px}
header h1{font-size:1.5rem;margin:0 0 6px;line-height:1.3}
.meta{color:var(--muted);font-size:.9rem;margin:0 0 4px}
.meta code{font-size:.85rem}
nav.toc{margin:18px 0 26px;padding:12px 16px;border:1px solid var(--line);border-radius:10px;background:var(--card);font-size:.95rem}
nav.toc a{color:var(--accent);text-decoration:none}
nav.toc ol{margin:6px 0 0;padding-left:20px}
h2.part{font-size:1.15rem;margin:36px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--line)}
.turn{border:1px solid var(--line);background:var(--card);border-radius:12px;padding:14px 20px;margin:14px 0;overflow-wrap:anywhere}
.turn.user{background:var(--user);border-color:var(--userline)}
.who{font-size:.75rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin-bottom:4px}
.md h1{font-size:1.3rem}.md h2{font-size:1.15rem}.md h3{font-size:1.02rem}
.md h1,.md h2,.md h3{margin:1.1em 0 .4em;line-height:1.35}
.md table{border-collapse:collapse;margin:10px 0;display:block;overflow-x:auto;font-size:.92rem}
.md th,.md td{border:1px solid var(--line);padding:4px 10px}
.md pre{background:var(--code);padding:10px 12px;border-radius:8px;overflow-x:auto;font-size:.85rem;line-height:1.45}
.md code{background:var(--code);padding:1px 4px;border-radius:4px;font-size:.88em}
.md pre code{background:none;padding:0}
.md hr{border:none;border-top:1px solid var(--line);margin:18px 0}
.katex-display{overflow-x:auto;overflow-y:hidden;padding:2px 0}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px;margin-top:12px}
figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px}
figure img{width:100%;height:auto;display:block;border-radius:4px}
figcaption{font-size:.8rem;color:var(--muted);text-align:center;padding-top:4px}
details.context{border:1px dashed var(--line);border-radius:12px;padding:6px 14px;margin:10px 0;background:transparent}
details.context>summary{cursor:pointer;font-weight:600;padding:6px 0;color:var(--accent)}
.note{font-size:.8rem;color:var(--muted);font-style:italic;margin:8px 0 0}
</style></head><body><main>
<header><h1>__TITLE__</h1>__META__</header>
<nav class="toc"><strong>Contents</strong><ol>__TOC__</ol></nav>
__BODY__
</main>
<script>
// Protect LaTeX from Markdown, render Markdown, then put back KaTeX output.
function render(el){
  const src = el.textContent, math = [];
  const keep = (tex, display) => { math.push({tex, display}); return `@@MATH${math.length-1}@@`; };
  let s = src.replace(/\\\\\\[([\\s\\S]+?)\\\\\\]/g, (_, t) => keep(t, true))
             .replace(/\\$\\$([\\s\\S]+?)\\$\\$/g, (_, t) => keep(t, true))
             .replace(/\\\\\\(([\\s\\S]+?)\\\\\\)/g, (_, t) => keep(t, false));
  let out = (window.marked ? marked.parse(s) : '<pre>'+s.replace(/</g,'&lt;')+'</pre>');
  out = out.replace(/@@MATH(\\d+)@@/g, (_, i) => {
    const m = math[+i];
    try { return katex.renderToString(m.tex, {displayMode: m.display, throwOnError: false}); }
    catch(e){ return m.tex; }
  });
  el.innerHTML = out;
}
document.querySelectorAll('.md').forEach(render);
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--context-turns", type=int, default=2)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()

    log = read_eval_log(a.log, resolve_attachments=True)
    s = log.samples[0]
    model = log.eval.model.split("/")[-1]
    effort = log.eval.model_generate_config.reasoning_effort
    msgs = list(s.messages)
    ctx, convo = msgs[: a.context_turns], msgs[a.context_turns:]
    unit = (s.metadata or {}).get("unit", str(s.id))
    title = a.title or f"{model}: follow-up conversation on {unit.split('/')[-1]}"

    meta = (f'<p class="meta">Model <code>{html.escape(log.eval.model)}</code> · reasoning effort '
            f'<code>{html.escape(str(effort))}</code> · no tools, no web search (verified from the raw API requests)</p>'
            f'<p class="meta">Unit <code>{html.escape(unit)}</code> · source log <code>{html.escape(a.log)}</code></p>')

    body, toc = [], []
    if ctx:
        toc.append('<li><a href="#context">Original task and answer (context the model saw)</a></li>')
        inner = "".join(render_turn(m, model, "Task (task.md + images)" if m.role == "user" else f"{model}: original answer")
                        for m in ctx)
        body.append(f'<h2 class="part" id="context">Context</h2><details class="context"><summary>Show the original '
                    f'task, images and answer</summary>{inner}</details>')
    q = 0
    for m in convo:
        if m.role == "user":
            q += 1
            first = parts(m)[0][0][1] if parts(m)[0] else ""
            short = first.strip().split("\n")[0]
            short = short[:90] + ("…" if len(short) > 90 else "")
            toc.append(f'<li><a href="#q{q}">Follow-up {q}: {html.escape(short)}</a></li>')
            body.append(f'<h2 class="part" id="q{q}">Follow-up {q}</h2>')
            body.append(render_turn(m, model, "Question"))
        else:
            body.append(render_turn(m, model, f"{model} (reasoning {effort})"))

    page = (PAGE.replace("__TITLE__", html.escape(title)).replace("__META__", meta)
            .replace("__TOC__", "".join(toc)).replace("__BODY__", "".join(body)))
    Path(a.out).write_text(page, encoding="utf-8")
    print(f"wrote {a.out} ({Path(a.out).stat().st_size/1e6:.1f} MB), {q} follow-up(s), {len(ctx)} context turn(s)")


if __name__ == "__main__":
    main()
