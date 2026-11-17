"""Extraction benchmark units (extractions/<paper>/<Tn_...>/): task.md + images -> model answer.

  inspect eval tasks/extraction.py -T unit=gan2025_star_ps_p2vp/T2_gisans_4arm_depth \
      --model openai/<model> --reasoning-effort high

What the model gets: ONE user message = task.md text followed by image1.png, image2.png, ...
(each preceded by its file name, which task.md refers to). Nothing else: no system prompt, no tools.
analysis.json / conclusion.json (ground truth) are never read here, not even into metadata, so they
cannot reach the model or the log's input. Grade the saved answers against them afterwards.

No online search, enforced three ways:
  1. harness is plain generate(): the request carries no tools, and provider-native search
     (e.g. OpenAI's web_search) only exists when passed as a tool.
  2. assert_no_tools() fails the sample if anything put a tool on the state before the call.
  3. scripts/verify_no_tools.py checks the raw API requests in the log: no `tools`, no search.
Sampling settings are left at the model's defaults (only what you pass on the CLI is set).
"""
from __future__ import annotations

import re
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser, ContentImage, ContentText
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver

ROOT = Path(__file__).resolve().parents[1] / "extractions"
TAG = "plain"
GROUND_TRUTH = {"analysis.json", "conclusion.json", "answer.md"}


def _image_key(p: Path) -> int:
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else 0


def drop_md_sections(text: str, names: list[str]) -> str:
    """Remove `## <name>` sections (heading through the line before the next `## ` heading)."""
    if not names:
        return text
    wanted = {n.strip().lower() for n in names if n.strip()}
    out, skipping = [], False
    for line in text.splitlines(keepends=True):
        if line.startswith("## "):
            skipping = line[3:].strip().lower() in wanted
        if not skipping:
            out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "".join(out))


def _names(x: str | list | None) -> list[str]:
    if not x:
        return []
    return [str(s) for s in x] if isinstance(x, list) else [s for s in str(x).split(",")]


def load_unit(unit: str, drop_sections: str | list | None = None) -> Sample:
    """drop_sections: task.md `## ` sections to leave out of the prompt, e.g. "Images"."""
    d = (ROOT / unit).resolve()
    if not (d / "task.md").exists():
        raise FileNotFoundError(f"{d}/task.md not found")
    images = sorted(d.glob("image*.png"), key=_image_key)
    if not images:
        raise FileNotFoundError(f"no image*.png in {d}")
    text = (d / "task.md").read_text(encoding="utf-8")
    dropped = _names(drop_sections)
    for name in dropped:
        if f"\n## {name.strip()}" not in "\n" + text:
            raise ValueError(f"task.md of {unit} has no section '## {name.strip()}'")
    content: list = [ContentText(text=drop_md_sections(text, dropped))]
    for img in images:
        assert img.name not in GROUND_TRUTH
        content += [ContentText(text=f"{img.name}:"), ContentImage(image=str(img), detail="high")]
    return Sample(
        id=unit.replace("/", "__"),
        input=[ChatMessageUser(content=content)],
        metadata={"unit": unit, "images": [i.name for i in images], "dropped_sections": dropped},
    )


@solver
def assert_no_tools() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if state.tools:
            raise RuntimeError(f"tools present but this run must have none: {[str(t) for t in state.tools]}")
        return state

    return solve


@task
def extraction(
    unit: str = "gan2025_star_ps_p2vp/T2_gisans_4arm_depth",
    judge: str | None = "openai/gpt-6.1-sol",
    judge_effort: str = "high",
    drop_sections: str | list | None = None,
) -> Task:
    """judge: model grading each ground-truth point after generation (labkit.judge); "none" to skip.
    Ground truth is read only by the scorer, after the answer is final.
    drop_sections: task.md sections to leave out of the prompt, e.g. -T drop_sections=Images."""
    units = [u.strip() for u in unit.split(",") if u.strip()]
    scorer = None
    if judge and str(judge).lower() != "none":
        from labkit.judge import gt_judge

        scorer = gt_judge(judge_model=judge, reasoning_effort=judge_effort,
                          tag=TAG + "".join(f"-no{n.strip()}" for n in _names(drop_sections)))
    return Task(
        dataset=[load_unit(u, drop_sections) for u in units],
        solver=[assert_no_tools(), generate()],
        scorer=scorer,
    )
