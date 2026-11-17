"""Ask a follow-up question in the same conversation as an earlier run.

  inspect eval tasks/followup.py --model openai/gpt-6-astra --reasoning-effort high \
      -T log=logs/extraction/<file>.eval -T question="..." --log-dir logs/followup

The saved conversation is replayed exactly as the model saw it (task.md + images, then the model's
own assistant turn, including its encrypted reasoning items, which OpenAI accepts back), and the
question is appended as a new user turn. Use the same model as the original run: encrypted
reasoning only decodes for the model/provider that produced it.

Same guarantees as tasks/extraction.py: plain generate(), no tools, no search; run
scripts/verify_no_tools.py on the log. No judge (there is no ground truth for a free question).
"""
from __future__ import annotations

from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import generate

from extraction import assert_no_tools

ROOT = Path(__file__).resolve().parents[1]


@task
def followup(log: str, question: str | list[str] | None = None, question_file: str | None = None,
             sample: str | None = None) -> Task:
    """question: the new user turn. Inspect splits -T values on commas, so a list is re-joined;
    for long or multi-line questions use question_file=path/to/question.md instead."""
    if question_file:
        qp = Path(question_file) if Path(question_file).is_absolute() else ROOT / question_file
        question = qp.read_text(encoding="utf-8").strip()
    elif isinstance(question, list):
        question = ", ".join(str(q) for q in question)
    if not question:
        raise ValueError("give question=... or question_file=...")
    path = Path(log) if Path(log).is_absolute() else ROOT / log  # Inspect runs tasks from tasks/
    src = read_eval_log(str(path), resolve_attachments=True)
    samples = [s for s in (src.samples or []) if sample is None or str(s.id) == sample]
    if not samples:
        raise ValueError(f"no sample {sample!r} in {log}")
    dataset = []
    for s in samples:
        history = list(s.messages)
        if history[-1].role != "assistant":
            raise ValueError(f"{s.id}: conversation does not end with the model's answer")
        dataset.append(Sample(
            id=f"{s.id}__followup",
            input=history + [ChatMessageUser(content=question)],
            metadata={**(s.metadata or {}), "source_log": log, "source_model": src.eval.model,
                      "question": question},
        ))
    return Task(dataset=dataset, solver=[assert_no_tools(), generate()])
