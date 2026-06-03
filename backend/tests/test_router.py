"""Router fallback + specialist name guard."""
from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_router_returns_default_on_llm_failure(monkeypatch):
    import graph.qa_agent_nodes as qan

    class BrokenLLM:
        def with_structured_output(self, schema):
            return self

        async def ainvoke(self, msgs, config=None):
            raise RuntimeError("simulated LLM failure")

    monkeypatch.setattr(qan, "_llm", lambda key="": BrokenLLM())

    state = {
        "session_id": "s",
        "repo_name": "r",
        "repo_summary": "",
        "architecture_notes": "",
        "repo_structure": [],
        "chat_history": [],
        "current_question": "what is this?",
        "current_answer": "",
        "selected_specialists": [],
        "specialist_outputs": {},
    }
    out = await qan.router(state)
    assert out["selected_specialists"] == ["code_reader"]


@pytest.mark.asyncio
async def test_router_filters_unknown_specialists(monkeypatch):
    import graph.qa_agent_nodes as qan
    from graph.qa_agent_nodes import _RouterDecision

    class FakeLLM:
        def with_structured_output(self, schema):
            return self

        async def ainvoke(self, msgs, config=None):
            return _RouterDecision(specialists=["code_reader", "bogus_agent", "arch_explainer"])

    monkeypatch.setattr(qan, "_llm", lambda key="": FakeLLM())

    out = await qan.router({
        "session_id": "s",
        "repo_name": "r",
        "repo_summary": "",
        "architecture_notes": "",
        "repo_structure": [],
        "chat_history": [],
        "current_question": "?",
        "current_answer": "",
        "selected_specialists": [],
        "specialist_outputs": {},
    })
    assert set(out["selected_specialists"]) == {"code_reader", "arch_explainer"}


def test_fan_out_emits_send_per_specialist():
    from graph.qa_agent_nodes import fan_out_to_specialists

    sends = fan_out_to_specialists({
        "selected_specialists": ["code_reader", "dep_tracer"],
        "current_question": "x",
        "session_id": "s",
    })
    names = [s.node for s in sends]
    assert names == ["code_reader", "dep_tracer"]


def test_fan_out_falls_back_when_empty():
    from graph.qa_agent_nodes import fan_out_to_specialists

    sends = fan_out_to_specialists({
        "selected_specialists": [],
        "current_question": "x",
        "session_id": "s",
    })
    assert len(sends) == 1 and sends[0].node == "code_reader"
