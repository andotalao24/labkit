"""Ground-truth judge for extraction units: grades a model answer point by point.

Every ground-truth point (analysis.json steps, per image and cross-image, plus conclusion.json
claims and extrapolations) gets exactly one verdict:
  right    1 point   the answer states this finding (same meaning; numbers within tolerance)
  wrong    0 points  the answer addresses the point but contradicts it, is out of tolerance, or
                     matches an alternative with credit < 1
  missing  (no points, listed separately)  the answer does not address the point

score = right / total points. accuracy_answered = right / (right + wrong).

Re-score an existing log (the model answer is not regenerated):
  inspect score logs/extraction/<file>.eval --scorer src/labkit/judge.py@gt_judge --action append \
      -S judge_model=openai/gpt-6.1-sol -S reasoning_effort=high
Reports are written to logs/extraction/judgements/<sample>__<answer model>.{md,json}.
The judge gets text only (task.md, the answer, the ground-truth points), no tools.
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, ResponseSchema, get_model
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import TaskState
from inspect_ai.util import json_schema

ROOT = Path(__file__).resolve().parents[2]
EXTRACTIONS = ROOT / "extractions"
REPORTS = ROOT / "logs" / "extraction" / "judgements"


# ---- ground truth -----------------------------------------------------------------------------
def load_points(unit_dir: Path) -> list[dict]:
    """Ground-truth points in a fixed order, each with what the judge needs to decide."""
    a = json.loads((unit_dir / "analysis.json").read_text())
    c = json.loads((unit_dir / "conclusion.json").read_text())
    params = {p["id"]: p for p in a.get("parameters", [])}

    def values(ids: list) -> list[dict]:
        out = []
        for v in ids:
            p = params.get(v) if isinstance(v, str) else v
            if p:
                out.append({k: p.get(k) for k in ("id", "name", "symbol", "value", "unit", "sample",
                                                  "reported_uncertainty", "tolerance", "scoring_condition")})
        return out

    points = []
    for group in a["per_image"]:
        for s in group["steps"]:
            points.append({
                "id": s["step_id"], "section": f"{s['image_id']} {s['stage']}", "aspect": s["aspect"],
                "ground_truth": {"observation": s["observation"], "interpretation": s["interpretation"]},
                "values": values(s.get("values", [])), "alternatives": s.get("alternatives", []),
            })
    for s in a.get("cross_image", []):
        imgs = s["image_id"] if isinstance(s["image_id"], list) else [s["image_id"]]
        points.append({
            "id": s["step_id"], "section": "cross " + ",".join(imgs), "aspect": s["aspect"],
            "ground_truth": {"observation": s["observation"], "interpretation": s["interpretation"]},
            "values": values(s.get("values", [])), "alternatives": s.get("alternatives", []),
        })
    for cl in c["nanostructure"]["claims"]:
        points.append({
            "id": cl["id"], "section": "conclusion (a) nanostructure", "aspect": cl["aspect"],
            "ground_truth": {"statement": cl["statement"]}, "values": values(cl.get("values", [])),
            "alternatives": [], "note": f"inferability: {cl.get('inferability')}",
        })
    for e in c.get("extrapolation", []):
        points.append({
            "id": e["id"], "section": "conclusion (b) extrapolation", "aspect": "extrapolation",
            "ground_truth": {"statement": e["statement"]}, "values": [], "alternatives": [],
        })
    return points


def batches(points: list[dict]) -> list[list[dict]]:
    """One judge call per image / cross-image / conclusion section (keeps each call focused)."""
    out: dict[str, list[dict]] = {}
    for p in points:
        key = p["section"].split()[0] if not p["section"].startswith("conclusion") else "conclusion"
        out.setdefault(key, []).append(p)
    return list(out.values())


# ---- judge call -------------------------------------------------------------------------------
class Verdict(BaseModel):
    id: str
    verdict: Literal["right", "wrong", "missing"]
    answer_quote: str          # the passage of the answer the verdict rests on ("" if missing)
    matched_alternative: str   # the alternative's `model` text if one matched, else ""
    reason: str


class Verdicts(BaseModel):
    verdicts: list[Verdict]


Verdict.model_rebuild()
Verdicts.model_rebuild(_types_namespace={"Verdict": Verdict})


SYSTEM = """You grade a scientific image-analysis answer against ground truth, one ground-truth point at a time.

For EACH ground-truth point give exactly one verdict:
- "right": the answer states this finding with the same scientific meaning. Wording may differ.
  Numbers must agree within the given tolerance; if no tolerance is given, within about 15%.
  If the ground truth says something is not applicable / not determinable, the answer is right
  if it also says so (or clearly declines to determine it).
- "wrong": the answer addresses this point but contradicts it, gives a value out of tolerance,
  asserts something the ground truth says cannot be determined, or matches an alternative whose
  credit is below 1. An alternative with credit 1 counts as right.
- "missing": the answer does not address this point at all.

Rules:
- Judge each point independently against the WHOLE answer (the finding may appear anywhere,
  e.g. a per-image point stated in the cross-image comparison counts).
- A point with several parts (observation + interpretation, or a statement with several facts) is
  right only if its essential content is there and none of it is contradicted. If the answer
  covers it only partly with nothing contradicted, decide by whether the core finding is stated.
- Hedging between the correct finding and an incompatible one is wrong unless the answer commits
  to the correct one as its main reading.
- Do not reward or penalise anything outside the ground-truth points.
- answer_quote: copy the shortest passage of the answer your verdict rests on ("" if missing).
- Return one verdict for every id given, using the ids exactly as given."""


def judge_prompt(task_md: str, answer: str, points: list[dict]) -> str:
    return (
        "## Task the answer responds to\n\n" + task_md
        + "\n\n## Answer to grade\n\n<answer>\n" + answer + "\n</answer>\n\n"
        + "## Ground-truth points to grade\n\n```json\n" + json.dumps(points, indent=1, ensure_ascii=False)
        + "\n```\n\nReturn a verdict for each of these ids: " + ", ".join(p["id"] for p in points)
    )


async def judge_batch(model, task_md: str, answer: str, points: list[dict], tries: int = 3) -> dict[str, dict]:
    todo, got = list(points), {}
    for _ in range(tries):
        if not todo:
            break
        out = await model.generate([ChatMessageSystem(content=SYSTEM),
                                    ChatMessageUser(content=judge_prompt(task_md, answer, todo))])
        text = out.completion.strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
        try:
            parsed = Verdicts.model_validate_json(text)
        except Exception:
            continue
        ids = {p["id"] for p in todo}
        for v in parsed.verdicts:
            if v.id in ids:
                got[v.id] = v.model_dump()
        todo = [p for p in todo if p["id"] not in got]
    for p in todo:  # the judge never returned these: record, do not guess
        got[p["id"]] = {"id": p["id"], "verdict": "ungraded", "answer_quote": "", "matched_alternative": "",
                        "reason": "judge returned no valid verdict"}
    return got


# ---- report -----------------------------------------------------------------------------------
def tally(rows: list[dict]) -> dict:
    n = {k: sum(r["verdict"] == k for r in rows) for k in ("right", "wrong", "missing", "ungraded")}
    total = len(rows)
    answered = n["right"] + n["wrong"]
    return {**n, "total": total, "score": n["right"] / total if total else 0.0,
            "accuracy_answered": n["right"] / answered if answered else 0.0}


def write_report(name: str, meta: dict, rows: list[dict]) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"{name}.json").write_text(json.dumps({**meta, "summary": tally(rows), "points": rows}, indent=1,
                                                     ensure_ascii=False))
    t = tally(rows)
    lines = [f"# Judgement: {meta['unit']}", "",
             f"- answer model: `{meta['answer_model']}`", f"- harness: `{meta.get('harness', 'unspecified')}`",
             f"- judge: `{meta['judge_model']}` (reasoning {meta['reasoning_effort']})",
             f"- **score: {t['right']}/{t['total']} = {t['score']:.3f}** (right 1, wrong 0; missing counted as not earned)",
             f"- right {t['right']} · wrong {t['wrong']} · missing {t['missing']}"
             + (f" · ungraded {t['ungraded']}" if t["ungraded"] else ""),
             f"- accuracy on points the answer addressed: {t['accuracy_answered']:.3f}", "",
             "## By section", "", "| section | right | wrong | missing | total |", "|---|---:|---:|---:|---:|"]
    sections: dict[str, list[dict]] = {}
    for r in rows:
        sections.setdefault(r["section"], []).append(r)
    for s, rs in sections.items():
        st = tally(rs)
        lines.append(f"| {s} | {st['right']} | {st['wrong']} | {st['missing']} | {st['total']} |")
    lines += ["", "## Points", "", "| id | section | aspect | verdict | point | reason |", "|---|---|---|---|---:|---|"]
    for r in rows:
        pt = {"right": "1", "wrong": "0"}.get(r["verdict"], "missing" if r["verdict"] == "missing" else "-")
        reason = r["reason"].replace("|", "\\|").replace("\n", " ")
        if r.get("matched_alternative"):
            reason += f" (matched alternative: {r['matched_alternative']})"
        lines.append(f"| {r['id']} | {r['section']} | {r['aspect']} | **{r['verdict']}** | {pt} | {reason} |")
    path = REPORTS / f"{name}.md"
    path.write_text("\n".join(lines) + "\n")
    return path


# ---- scorer -----------------------------------------------------------------------------------
@scorer(metrics={"score": [mean()], "right": [mean()], "wrong": [mean()], "missing": [mean()],
                 "accuracy_answered": [mean()]})
def gt_judge(judge_model: str = "openai/gpt-6.1-sol", reasoning_effort: str = "high", tag: str = ""):
    """tag: harness/run label added to the report file name (e.g. "plain", "codex") so runs of the
    same model on the same unit do not overwrite each other's reports."""
    async def score(state: TaskState, target: Target) -> Score:
        unit = state.metadata["unit"]
        unit_dir = EXTRACTIONS / unit
        task_md = (unit_dir / "task.md").read_text(encoding="utf-8")
        answer = state.output.completion
        points = load_points(unit_dir)
        model = get_model(judge_model, config=GenerateConfig(
            reasoning_effort=reasoning_effort,
            response_schema=ResponseSchema(name="verdicts", json_schema=json_schema(Verdicts), strict=False)))  # strict rejects Inspect's nested schema; validated below
        results: dict[str, dict] = {}
        for got in await asyncio.gather(*(judge_batch(model, task_md, answer, b) for b in batches(points))):
            results.update(got)
        rows = [{**{k: p[k] for k in ("id", "section", "aspect")}, **results[p["id"]]} for p in points]
        t = tally(rows)
        # subscription runs use --model none; the solver records the CLI's actual model
        answer_model = state.metadata.get("answer_model") or str(state.model)
        name = f"{state.sample_id}__{answer_model.replace('/', '_')}" + (f"__{tag}" if tag else "")
        report = write_report(name, {"unit": unit, "answer_model": answer_model, "harness": tag or "unspecified",
                                     "judge_model": judge_model, "reasoning_effort": reasoning_effort}, rows)
        return Score(
            value={k: t[k] for k in ("score", "right", "wrong", "missing", "accuracy_answered")},
            answer=answer[:200],
            explanation=f"right {t['right']}, wrong {t['wrong']}, missing {t['missing']} of {t['total']} "
                        f"(ungraded {t['ungraded']}); report: {report}",
            metadata={"points": rows, "judge_model": judge_model, "report": str(report)},
        )

    return score
