from langgraph.graph import StateGraph, END
from graph.state import RepoState
from graph.ingest_nodes import (
    ingest_repo,
    summarize_repo_summary,
    summarize_architecture,
    join_summaries,
)
from graph.qna_nodes import (
    retrieve_context,
    build_prompt,
    llm_call,
    execute_tool,
    synthesise_answer,
    update_history,
    route_after_llm,
    route_after_tool,
)


def build_ingest_graph():
    """
    ingest_repo ──┬──> summarize_repo_summary ──┬──> join_summaries ──> END
                  └──> summarize_architecture ──┘
    """
    graph = StateGraph(RepoState)
    
    # Add nodes
    graph.add_node("ingest", ingest_repo)
    graph.add_node("summarize_summary", summarize_repo_summary)
    graph.add_node("summarize_architecture", summarize_architecture)
    graph.add_node("join", join_summaries)
    
    # Set entry point
    graph.set_entry_point("ingest")
    
    # Fan-out parallel edges
    graph.add_edge("ingest", "summarize_summary")
    graph.add_edge("ingest", "summarize_architecture")
    
    # Fan-in join edges
    graph.add_edge("summarize_summary", "join")
    graph.add_edge("summarize_architecture", "join")
    
    # End edge
    graph.add_edge("join", END)
    
    return graph.compile()


def build_qa_graph():
    """
    retrieve_context ──> build_prompt ──> llm_call ── (route) ──> synthesise_answer ──> update_history ──> END
                                             ▲                  │
                                             └──── execute_tool ┘
    """
    graph = StateGraph(RepoState)
    
    # Add nodes
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("build_prompt", build_prompt)
    graph.add_node("llm_call", llm_call)
    graph.add_node("execute_tool", execute_tool)
    graph.add_node("synthesise_answer", synthesise_answer)
    graph.add_node("update_history", update_history)
    
    # Set entry point
    graph.set_entry_point("retrieve_context")
    
    # Linear edges
    graph.add_edge("retrieve_context", "build_prompt")
    graph.add_edge("build_prompt", "llm_call")
    
    # ReAct Routing: after llm_call, conditionally go to execute_tool OR synthesise_answer
    graph.add_conditional_edges(
        "llm_call",
        route_after_llm,
        {
            "execute_tool": "execute_tool",
            "synthesise_answer": "synthesise_answer",
        }
    )
    
    # Loop path: after execute_tool, loop back to llm_call
    graph.add_conditional_edges(
        "execute_tool",
        route_after_tool,
        {
            "llm_call": "llm_call"
        }
    )
    
    # Final answer & cleanup
    graph.add_edge("synthesise_answer", "update_history")
    graph.add_edge("update_history", END)
    
    return graph.compile()


# Module-level compiled graphs — imported directly by main.py
ingest_graph = build_ingest_graph()
qa_graph = build_qa_graph()
