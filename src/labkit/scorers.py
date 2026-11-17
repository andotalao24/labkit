"""tool_usage(): add to any task's scorers to record what the agent actually called.

Reports per sample: tool_calls (all), search_calls (our/Inspect's web_search tool), and an
explanation with the per-tool counts. Provider-native search (search=native) runs server side and
shows up as ContentToolUse blocks in the assistant message, counted as native_search_calls.
"""
from __future__ import annotations

import json
from collections import Counter

from inspect_ai.model import ChatMessageAssistant
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import TaskState


@scorer(metrics={"tool_calls": [mean()], "search_calls": [mean()], "native_search_calls": [mean()]})
def tool_usage():
    async def score(state: TaskState, target: Target) -> Score:
        counts: Counter[str] = Counter()
        native = 0
        for m in state.messages:
            if not isinstance(m, ChatMessageAssistant):
                continue
            for tc in m.tool_calls or []:
                counts[tc.function] += 1
            if isinstance(m.content, list):
                native += sum(1 for c in m.content if getattr(c, "type", None) == "tool_use")
        return Score(
            value={"tool_calls": sum(counts.values()), "search_calls": counts.get("web_search", 0),
                   "native_search_calls": native},
            explanation=json.dumps(dict(counts)),
        )

    return score
