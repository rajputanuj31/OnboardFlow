import sys
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

# Ensure backend root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from graph.state import RepoState
from utils.vector_db import query_vector_db
from graph.helpers import _llm, _build_file_contents, _read_codebase_file

MAX_REACT_ITERATIONS = 5

@tool
def read_codebase_file(filepath: str) -> str:
    """
    Read the contents of a specific file in the repository codebase.
    filepath should be the relative path of the file from the repository root (e.g. 'src/utils/api.ts').
    """
    return ""


async def retrieve_context(state: RepoState) -> dict:
    """Query SQLite vector DB; return top-k RAG chunks for current_question."""
    chunks = await query_vector_db(
        state["session_id"],
        state["current_question"],
        top_k=4,
        model_api_key=state.get("model_api_key", "")
    )
    rag_snippets = []
    for chunk in chunks:
        rag_snippets.append(
            f"=== {chunk['filepath']} (Chunk similarity: {chunk['similarity']:.4f}) ===\n{chunk['content']}"
        )
    rag_context = "\n\n".join(rag_snippets) if rag_snippets else "(No highly relevant chunks found.)"
    
    return {"rag_context": rag_context}


def build_prompt(state: RepoState) -> dict:
    """Assemble static system msg (repo summary, arch, file contents)
    and dynamic system msg (RAG snippets) + inject chat_history."""

    # 1. Build priority file contents block
    all_files: dict[str, str] = {}
    if state.get("readme"):
        all_files["README"] = state["readme"][:4000]
    if state.get("contributing"):
        all_files["CONTRIBUTING"] = state["contributing"][:2000]
    all_files.update(state.get("fetched_files", {}))
    file_contents = _build_file_contents(all_files, char_limit=8_000)
    structure_str = "\n".join(state.get("repo_structure", []))

    # 2. Static prompt instructions
    static_system_text = f"""You are an expert on the "{state['repo_name']}" GitHub repository.

Here is everything you know about it:

## Summary
{state['repo_summary']}

## Architecture
{state['architecture_notes']}

## Repository Directory Layout
{structure_str}

## Key File Contents (Static Priority Files)
{file_contents}

CRITICAL RULES:
1. You MUST NOT use your general knowledge to answer questions about the repository's features, logic, or implementation.
2. Answer based strictly on the repository content and the files you fetch.
3. DO NOT hallucinate, guess, or make up code snippets, routes, file names, or application logic.
4. If the user's question asks about how a feature works (e.g., "how to upload a video", "what is the flow"), you MUST FIRST identify the relevant files from the 'Repository Directory Layout' and call the `read_codebase_file` tool to read them. DO NOT give a generic explanation of how such a feature typically works.
5. You are strictly forbidden from writing example or placeholder code (like generic express, django, or react handlers) unless they are verbatim present in the context. If you cannot find the actual code, use the tool to fetch it.
6. If the files cannot be found or read, state clearly that you cannot find the implementation.
"""

    # 3. Dynamic RAG context
    dynamic_system_text = f"""## Relevant Code Snippets (Semantic Search RAG)
Here are relevant snippets from the codebase that might contain the answer or point to the correct files:

{state.get('rag_context', '')}
"""

    messages = [
        SystemMessage(content=static_system_text),
        SystemMessage(content=dynamic_system_text)
    ]

    # 4. Inject history (up to last 6 messages)
    for msg in state.get("chat_history", [])[-6:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    # 5. Append current question
    messages.append(HumanMessage(content=state["current_question"]))

    return {"messages": messages}


async def llm_call(state: RepoState) -> dict:
    """Invoke ChatOpenAI with bound tools. Sets needs_tool_call and
    increments tool_calls_made if a tool_call is present in the response."""
    api_key = state.get("model_api_key", "")
    llm_with_tools = _llm(api_key).bind_tools([read_codebase_file])
    
    # Invoke LLM with current message sequence
    response = await llm_with_tools.ainvoke(state["messages"])
    
    # Determine if tool execution is required
    has_tool_call = bool(response.tool_calls)
    
    return {
        "messages": state["messages"] + [response],
        "needs_tool_call": has_tool_call,
    }


async def execute_tool(state: RepoState) -> dict:
    """Run read_codebase_file(): check fetched_files → SQLite chunks → GitHub API.
    Appends ToolMessage to message list; bumps tool_calls_made."""
    last_msg = state["messages"][-1]
    tool_messages = []
    
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        for tool_call in last_msg.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            
            if tool_name == "read_codebase_file":
                filepath = tool_args.get("filepath")
                print(f"[REACT] Tool Call: read_codebase_file('{filepath}')")
                
                # Execute helper
                tool_output = await _read_codebase_file(filepath, state)
                
                tool_messages.append(
                    ToolMessage(content=str(tool_output), tool_call_id=tool_id)
                )
            else:
                tool_messages.append(
                    ToolMessage(content=f"Error: Unknown tool '{tool_name}'", tool_call_id=tool_id)
                )
                
    return {
        "messages": state["messages"] + tool_messages,
        "tool_calls_made": state.get("tool_calls_made", 0) + 1,
        "needs_tool_call": False,
    }


async def synthesise_answer(state: RepoState) -> dict:
    """Final LLM pass — synthesise answer from accumulated tool results.
    Only reached when LLM returned no tool_call, or MAX_REACT_ITERATIONS hit."""
    print("[REACT] Final invocation to synthesize answer...")
    api_key = state.get("model_api_key", "")
    
    response = await _llm(api_key).ainvoke(state["messages"])
    return {"current_answer": response.content}


def update_history(state: RepoState) -> dict:
    """Append {role:user, content:question} + {role:assistant, content:answer}
    to chat_history; reset tool_calls_made to 0."""
    updated_history = list(state.get("chat_history", [])) + [
        {"role": "user", "content": state["current_question"]},
        {"role": "assistant", "content": state["current_answer"]},
    ]
    return {
        "chat_history": updated_history,
        "tool_calls_made": 0
    }


# ── routing functions (conditional edges) ───────────────────────────────────

def route_after_llm(state: RepoState) -> str:
    """After llm_call: loop to execute_tool OR proceed to synthesise_answer."""
    iterations = state.get("tool_calls_made", 0)
    if state.get("needs_tool_call") and iterations < MAX_REACT_ITERATIONS:
        return "execute_tool"           # ReAct: run the tool
    return "synthesise_answer"          # done iterating — write final answer


def route_after_tool(state: RepoState) -> str:
    """After execute_tool: loop back to llm_call for next ReAct iteration."""
    return "llm_call" 
