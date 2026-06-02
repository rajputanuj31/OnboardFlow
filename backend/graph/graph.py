from langgraph.graph import END, START, StateGraph

from graph.ingest_nodes import (
    ingest_repo,
    summarize_repo_summary,
    summarize_architecture,
    join_summaries,
)
from graph.state import PRReviewState, QAState, RepoState


# ── Ingest graph (unchanged from v1) ──────────────────────────────────────────

def build_ingest_graph():
    """
    ingest_repo ──┬──> summarize_repo_summary ──┬──> join_summaries ──> END
                  └──> summarize_architecture ──┘
    """
    graph = StateGraph(RepoState)

    graph.add_node("ingest", ingest_repo)
    graph.add_node("summarize_summary", summarize_repo_summary)
    graph.add_node("summarize_architecture", summarize_architecture)
    graph.add_node("join", join_summaries)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "summarize_summary")
    graph.add_edge("ingest", "summarize_architecture")
    graph.add_edge("summarize_summary", "join")
    graph.add_edge("summarize_architecture", "join")
    graph.add_edge("join", END)

    return graph.compile()


# ── Q&A graph (v2 multi-agent) ────────────────────────────────────────────────

def build_qa_graph():
    """
    router ─┬─> code_reader ──────┐
            ├─> arch_explainer ───┼─> synthesizer ──> update_history ──> END
            └─> dep_tracer ───────┘
    """
    # Imported here to avoid pulling MCP/LangGraph prebuilt deps at module import
    # time (keeps `ingest_graph` instantiable even before MCP is wired).
    from graph.qa_agent_nodes import (
        arch_explainer,
        code_reader,
        dep_tracer,
        fan_out_to_specialists,
        router,
        synthesizer,
        update_history,
        SPECIALIST_NAMES,
    )

    graph = StateGraph(QAState)

    graph.add_node("router", router)
    graph.add_node("code_reader", code_reader)
    graph.add_node("arch_explainer", arch_explainer)
    graph.add_node("dep_tracer", dep_tracer)
    graph.add_node("synthesizer", synthesizer)
    graph.add_node("update_history", update_history)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", fan_out_to_specialists, list(SPECIALIST_NAMES))

    for name in SPECIALIST_NAMES:
        graph.add_edge(name, "synthesizer")

    graph.add_edge("synthesizer", "update_history")
    graph.add_edge("update_history", END)

    return graph.compile()


# ── PR review graph (v2 — Phase 4) ────────────────────────────────────────────

def build_pr_review_graph():
    """
    fetch_pr ─┬─> code_explainer ─┐
              ├─> impact_analyst ─┼─> synthesizer ──> END
              └─> risk_assessor ──┘
    """
    from graph.pr_review_nodes import (
        code_explainer,
        fetch_pr,
        impact_analyst,
        risk_assessor,
        synthesizer,
        PR_SPECIALIST_NAMES,
    )

    graph = StateGraph(PRReviewState)

    graph.add_node("fetch_pr", fetch_pr)
    graph.add_node("code_explainer", code_explainer)
    graph.add_node("impact_analyst", impact_analyst)
    graph.add_node("risk_assessor", risk_assessor)
    graph.add_node("synthesizer", synthesizer)

    graph.add_edge(START, "fetch_pr")
    for name in PR_SPECIALIST_NAMES:
        graph.add_edge("fetch_pr", name)
        graph.add_edge(name, "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()


# Module-level compiled graphs — imported directly by main.py.
ingest_graph = build_ingest_graph()
qa_graph = build_qa_graph()
pr_review_graph = build_pr_review_graph()
