import asyncio
import sys
from functools import lru_cache
from pathlib import Path

# Ensure backend root is on path when this module is imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_openai import ChatOpenAI
# pyrefly: ignore [missing-import]
from graph.state import RepoState
from onboard_github import fetch_repo_details
from prompts import SUMMARIZE_REPO_PROMPT, ARCHITECTURE_PROMPT, ANSWER_QUESTION_PROMPT


@lru_cache(maxsize=1)
def _llm() -> ChatOpenAI:
    """Lazy-initialize the LLM so load_dotenv() in main.py runs first."""
    return ChatOpenAI(model="gpt-4o-mini", max_tokens=1024)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_file_contents(files: dict[str, str], char_limit: int = 12_000) -> str:
    """Concatenate fetched files into a single string, respecting a char limit."""
    parts = []
    total = 0
    for filename, content in files.items():
        block = f"\n=== {filename} ===\n{content}"
        if total + len(block) > char_limit:
            break
        parts.append(block)
        total += len(block)
    return "".join(parts)


# ── Nodes ─────────────────────────────────────────────────────────────────────

async def ingest_repo(state: RepoState) -> dict:
    """Fetch all repository data from GitHub and populate state."""
    print(f"\n[INGEST] Fetching {state['repo_url']} ...")

    details = await fetch_repo_details(state["repo_url"].strip().rstrip("/"))

    print(f"[INGEST] {details.repo} — {details.repo_stars}★  {details.repo_language}")
    print(f"[INGEST] Files: {list(details.files.keys())}")

    return {
        "repo_name": details.repo,
        "repo_description": details.description,
        "repo_language": details.repo_language,
        "repo_stars": details.repo_stars,
        "repo_license": details.license,
        "repo_topics": details.topics,
        "repo_structure": details.structure,
        "readme": details.readme,
        "contributing": details.contributing,
        "fetched_files": details.files,
    }


async def summarize_repo(state: RepoState) -> dict:
    """Generate repo summary and architecture notes in parallel."""
    print(f"\n[SUMMARIZE] Analyzing {state['repo_name']} ...")

    structure_str = "\n".join(state["repo_structure"])

    # Include readme + contributing in the file contents for summarization
    all_files: dict[str, str] = {}
    if state["readme"]:
        all_files["README"] = state["readme"]
    if state["contributing"]:
        all_files["CONTRIBUTING"] = state["contributing"]
    all_files.update(state["fetched_files"])

    file_contents_full = _build_file_contents(all_files, char_limit=12_000)
    file_contents_short = _build_file_contents(all_files, char_limit=6_000)

    summary_prompt = SUMMARIZE_REPO_PROMPT.format(
        repo_name=state["repo_name"],
        repo_description=state["repo_description"],
        repo_language=state["repo_language"],
        repo_stars=state["repo_stars"],
        topics=", ".join(state["repo_topics"]),
        structure=structure_str,
        file_contents=file_contents_full,
    )

    arch_prompt = ARCHITECTURE_PROMPT.format(
        repo_name=state["repo_name"],
        repo_language=state["repo_language"],
        structure=structure_str,
        file_contents=file_contents_short,
    )

    # Run both LLM calls in parallel
    summary_response, arch_response = await asyncio.gather(
        _llm().ainvoke(summary_prompt),
        _llm().ainvoke(arch_prompt),
    )

    print("[SUMMARIZE] Done")

    return {
        "repo_summary": summary_response.content,
        "architecture_notes": arch_response.content,
    }


async def answer_question(state: RepoState) -> dict:
    """Answer a user question using session context."""
    print(f"\n[ANSWER] {state['current_question']}")

    # Last 3 exchanges (6 messages) to keep context tight
    history_str = ""
    for msg in state.get("chat_history", [])[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n\n"

    all_files: dict[str, str] = {}
    if state["readme"]:
        all_files["README"] = state["readme"][:4000]
    if state["contributing"]:
        all_files["CONTRIBUTING"] = state["contributing"][:2000]
    all_files.update(state["fetched_files"])

    file_contents = _build_file_contents(all_files, char_limit=8_000)

    prompt = ANSWER_QUESTION_PROMPT.format(
        repo_name=state["repo_name"],
        repo_summary=state["repo_summary"],
        architecture_notes=state["architecture_notes"],
        file_contents=file_contents,
        chat_history=history_str or "(no prior conversation)",
        question=state["current_question"],
    )

    response = await _llm().ainvoke(prompt)

    updated_history = list(state.get("chat_history", [])) + [
        {"role": "user", "content": state["current_question"]},
        {"role": "assistant", "content": response.content},
    ]

    print("[ANSWER] Done")

    return {
        "current_answer": response.content,
        "chat_history": updated_history,
    }
