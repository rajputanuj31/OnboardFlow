from graph.state import RepoState
from onboard_github import fetch_repo_details, parse_github_repo_url
from prompts import SUMMARIZE_REPO_PROMPT, ARCHITECTURE_PROMPT
from utils.vector_db import save_repo_chunks
from graph.helpers import _llm, _build_file_contents


# ── stub nodes ──────────────────────────────────────────────────────────────

async def ingest_repo(state: RepoState) -> dict:
    """Fetch repo metadata, files, README; embed + store chunks in vector DB."""

    #parse url and fetch data
    repo_url = state['repo_url']
    repo_path = parse_github_repo_url(repo_url)
    fetched_data = await fetch_repo_details(repo_path)

    #embed + store

    await save_repo_chunks(
        session_id=state['session_id'],
        files=fetched_data.files,
        readme=fetched_data.readme,
        contributing=fetched_data.contributing,
        model_api_key=state.get('model_api_key', ""),
    )
    
    #return metadata
    return {
        "repo_name": fetched_data.repo,
        "repo_description": fetched_data.description,
        "repo_language": fetched_data.repo_language,
        "repo_stars": fetched_data.repo_stars,
        "repo_license": fetched_data.license,
        "repo_topics": fetched_data.topics,
        "repo_structure": fetched_data.structure,
        "readme": fetched_data.readme,
        "contributing": fetched_data.contributing,
        "fetched_files": fetched_data.files,
    }

async def summarize_repo_summary(state: RepoState) -> dict:
    """LLM call A — generate plain-language repo summary (SUMMARIZE_REPO_PROMPT)."""

    #collect files
    all_files: dict[str, str] = {}
    if state.get("readme"):
        all_files["README.md"] = state["readme"]
    if state.get("contributing"):
        all_files["CONTRIBUTING.md"] = state["contributing"]
    all_files.update(state.get("fetched_files", {}))

    #build file contents
    file_contents = _build_file_contents(all_files, char_limit=12_000)
    structured_str = "\n".join(state["repo_structure"])
    
    #build the Prompt
    prompt = SUMMARIZE_REPO_PROMPT.format(
        repo_name=state['repo_name'],
        repo_description=state['repo_description'],
        repo_language=state['repo_language'],
        repo_stars=state['repo_stars'],
        topics=", ".join(state["repo_topics"]),
        structure=structured_str,
        file_contents=file_contents,
    )
    
    #call LLM A
    model_api_key = state.get("model_api_key", "")
    summary_resp = await _llm(model_api_key).ainvoke(prompt)
    return {"repo_summary": summary_resp.content}

async def summarize_architecture(state: RepoState) -> dict:
    """LLM call B — generate architecture notes (ARCHITECTURE_PROMPT)."""
    #collect files
    all_files: dict[str, str] = {}
    if state.get("readme"):
        all_files["README.md"] = state["readme"]
    if state.get("contributing"):
        all_files["CONTRIBUTING.md"] = state["contributing"]
    all_files.update(state.get("fetched_files", {}))

    #build file contents
    file_contents = _build_file_contents(all_files, char_limit=12_000)
    structured_str = "\n".join(state["repo_structure"])
    
    #build the Prompt
    prompt = ARCHITECTURE_PROMPT.format(
        repo_name=state['repo_name'],
        repo_language=state['repo_language'],
        structure=structured_str,
        file_contents=file_contents,
    )
    
    #call LLM A
    model_api_key = state.get("model_api_key", "")
    arch_resp = await _llm(model_api_key).ainvoke(prompt)
    return {"architecture_notes": arch_resp.content}

def join_summaries(state: RepoState) -> dict:
    """Fan-in: both LLM results are now in state; nothing extra to do."""
    return {}
    