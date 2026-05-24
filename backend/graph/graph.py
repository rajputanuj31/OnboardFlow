from langgraph.graph import StateGraph, END
# pyrefly: ignore [missing-import]
from graph.state import RepoState
# pyrefly: ignore [missing-import]
from graph.nodes import ingest_repo, summarize_repo, answer_question


def build_ingest_graph():
    """ingest → summarize → END"""
    graph = StateGraph(RepoState)
    graph.add_node("ingest", ingest_repo)
    graph.add_node("summarize", summarize_repo)
    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()


def build_qa_graph():
    """answer → END"""
    graph = StateGraph(RepoState)
    graph.add_node("answer", answer_question)
    graph.set_entry_point("answer")
    graph.add_edge("answer", END)
    return graph.compile()


# Module-level compiled graphs — imported directly by main.py
ingest_graph = build_ingest_graph()
qa_graph = build_qa_graph()
