"""
Prompt templates for the OSS onboarding agent.

All prompts use plain str.format() — no LangChain PromptTemplate needed.
"""

# ── Summarize node ────────────────────────────────────────────────────────────

SUMMARIZE_REPO_PROMPT = """\
You are analyzing a GitHub repository to help developers understand it quickly.

Repository: {repo_name}
Description: {repo_description}
Primary Language: {repo_language}
Stars: {repo_stars}
Topics: {topics}

Directory Structure:
{structure}

File Contents:
{file_contents}

Write a clear, practical summary covering:
1. What this project does (2-3 sentences)
2. Who it is for (end users, library consumers, contributors)
3. Main components and what each one does
4. Key technologies and dependencies
5. How to get started (setup, install, first run)

Rules:
- Be specific — use actual file names, class names, and module names from the content above
- Under 400 words
- No fluff
"""

ARCHITECTURE_PROMPT = """\
Based on the repository structure and files below, describe the architecture concisely.

Repository: {repo_name}
Language: {repo_language}

Directory Structure:
{structure}

File Contents:
{file_contents}

Provide:
1. Overall architectural pattern (MVC, library, CLI tool, microservice, etc.)
2. Main entry points and module layout
3. How the key modules/packages relate to each other
4. A visual architecture diagram represented as a Mermaid flowchart, wrapped strictly inside a single ```mermaid code block.

Rules:
- Keep the textual description under 250 words total, utilizing clean bullet points
- Ensure the Mermaid diagram is valid, use `graph TD` or `graph LR` layouts, and enclose any node labels in double quotes (e.g. A["File Name"]) to prevent rendering failures
- Only reference files/modules that are actually present above
"""

# ── Answer node ───────────────────────────────────────────────────────────────

ANSWER_QUESTION_PROMPT = """\
You are an expert on the "{repo_name}" GitHub repository.

Here is everything you know about it:

## Summary
{repo_summary}

## Architecture
{architecture_notes}

## Key File Contents
{file_contents}

## Conversation So Far
{chat_history}

## User's Question
{question}

Answer based strictly on the repository content above.
- Be specific — reference actual files, functions, and class names by name
- Include small code snippets when they add clarity
- If the answer requires a file you don't have, say exactly which file would help
- Keep answers focused and practical (no filler)
"""


# ── Multi-agent Q&A (v2) ──────────────────────────────────────────────────────

ROUTER_PROMPT = """\
You are a router deciding which specialist agents should answer a question about \
the "{repo_name}" repository.

Specialists available:
- code_reader: best for questions about specific functions, files, or implementations \
  ("how does X work", "show me the auth flow", "what does foo() do")
- arch_explainer: best for high-level questions about overall design, layout, or architecture \
  ("what's the architecture", "how is the project organized", "what are the main components")
- dep_tracer: best for questions about cross-file relationships, callers, dependencies \
  ("where is Y called", "what uses Z", "what would break if I change W")

## Repository context
Summary: {repo_summary}
Top-level structure (first lines): {structure_preview}

## Conversation so far
{chat_history}

## User's question
{question}

Pick between 1 and 3 specialists. A single specialist is usually enough; pick more \
only if the question genuinely spans multiple concerns.

Respond ONLY with a JSON object of the form:
{{"specialists": ["code_reader"]}}
or
{{"specialists": ["arch_explainer", "dep_tracer"]}}
"""

CODE_READER_PROMPT = """\
You are the **code_reader** specialist for the "{repo_name}" repository.
Your job is to answer file- and function-level questions by reading actual code.

You have these MCP tools available:
- read_codebase_file(session_id, filepath): fetch a file's contents
- semantic_search(session_id, query, top_k): find relevant chunks

Rules:
- Always use semantic_search first if you don't know which files to read.
- Then read the most relevant files via read_codebase_file before answering.
- Quote short snippets directly from the file. Never invent code.
- If the question is about overall architecture, say "outside my scope: arch_explainer".
- Be terse: 4–10 sentences, plus snippets only when they clarify.

Session ID for tool calls: {session_id}

Repository summary (for orientation only — do NOT answer from this alone):
{repo_summary}

Question: {question}
"""

ARCH_EXPLAINER_PROMPT = """\
You are the **arch_explainer** specialist for the "{repo_name}" repository.
Your job is to explain how the project is organized at a high level.

You have these MCP tools available:
- get_file_summary(session_id): the LLM-generated repo summary + Mermaid architecture
- get_repo_structure(session_id): the cleaned directory tree

Rules:
- Use get_file_summary and get_repo_structure to ground your answer.
- Describe modules, their responsibilities, and how they fit together.
- Reference real directory and file names.
- Do NOT dive into specific function implementations — that's code_reader's job.
- Be terse: 4–10 sentences. Prefer bullets.

Session ID for tool calls: {session_id}

Question: {question}
"""

DEP_TRACER_PROMPT = """\
You are the **dep_tracer** specialist for the "{repo_name}" repository.
Your job is to trace cross-file relationships: callers of a function, users of a module,
or the blast radius of a change.

You have these MCP tools available:
- semantic_search(session_id, query, top_k): primary tool — search for callers / usages
- read_codebase_file(session_id, filepath): confirm specific usages by reading the file

Rules:
- Start with semantic_search using the symbol name and likely caller phrasing.
- Read files that look like real usage sites; ignore unrelated matches.
- Be concrete: list specific files / line patterns where the dependency shows up.
- If unsure whether a match is real, say so — don't overclaim.
- Be terse: 4–10 sentences plus a short list of files/symbols touched.

Session ID for tool calls: {session_id}

Question: {question}
"""

QA_SYNTHESIZER_PROMPT = """\
You are the synthesizer for a multi-agent Q&A system answering questions about \
the "{repo_name}" repository.

Below are findings from one or more specialist agents. Combine them into ONE \
clean markdown answer for the user. Do not mention the agents by name. \
Do not repeat the same fact from multiple specialists. If they disagree, prefer \
the answer that cites specific files/code.

If only one specialist responded, just polish their answer; do not pad.

## Specialist findings
{specialist_outputs}

## Conversation so far
{chat_history}

## User's question
{question}

Write the final answer now. Use markdown. Be specific and concise.
"""


# ── PR review (v2 — Phase 4) ──────────────────────────────────────────────────

PR_CODE_EXPLAINER_PROMPT = """\
You are the **code_explainer** specialist reviewing a GitHub pull request.

Your job: explain in plain language WHAT this PR changes and WHY it appears to make those changes.

You have these MCP tools available:
- read_codebase_file(session_id, filepath): read surrounding context on touched files
- semantic_search(session_id, query, top_k): find related code

## PR metadata
Title: {pr_title}
Author: {pr_author}
Base → Head: {pr_base} → {pr_head}
Description:
{pr_description}

## Changed files
{pr_files_summary}

## Diff (truncated if long)
```diff
{pr_diff}
```

Rules:
- Read the touched files for context (especially if the diff is small but cross-cutting).
- Identify the purpose: bug fix, feature, refactor, perf, docs, etc.
- Call out anything that doesn't quite match the PR title/description.
- Be concrete: name the actual functions/classes changed.
- Output 4–8 bullets. No padding.

Session ID for tool calls (may be empty if no repo was pre-ingested): {session_id}
"""

PR_IMPACT_ANALYST_PROMPT = """\
You are the **impact_analyst** specialist reviewing a GitHub pull request.

Your job: identify which other parts of the codebase are LIKELY affected by these changes.

You have these MCP tools available:
- semantic_search(session_id, query, top_k): find callers of touched symbols
- get_repo_structure(session_id): the cleaned directory tree
- read_codebase_file(session_id, filepath): confirm specific usages

## PR metadata
Title: {pr_title}

## Changed files
{pr_files_summary}

## Diff (truncated if long)
```diff
{pr_diff}
```

Rules:
- For each touched public symbol (function, class, type, env var, route), search for its callers.
- Flag any tests that look stale or missing for the changes made.
- Identify ripple effects: shared utilities, config changes, API contract changes, schema migrations.
- Be honest about uncertainty: if you can't find callers, say so.
- Output 4–8 bullets.

Session ID for tool calls (may be empty if no repo was pre-ingested): {session_id}
"""

PR_RISK_ASSESSOR_PROMPT = """\
You are the **risk_assessor** specialist reviewing a GitHub pull request.

Your job: find correctness and security risks in the diff.

You have this MCP tool available:
- read_codebase_file(session_id, filepath): read surrounding context

## PR metadata
Title: {pr_title}

## Diff (truncated if long)
```diff
{pr_diff}
```

Risk checklist — go through each, only mention if you find something:
- Input validation / injection (SQL, command, path traversal)
- Auth / authorization bypass, missing permission checks
- Secret leakage (logs, error messages, commit content)
- Race conditions, concurrency, atomicity
- Error handling: swallowed exceptions, missing fallback, broken retries
- Performance footguns: N+1 queries, unbounded loops, blocking I/O on hot path
- Backwards compatibility: schema, API contract, env var name changes
- Test coverage gaps for the risky parts

Rules:
- Be specific: cite the file + line range when calling out a risk.
- Don't invent risks. If the diff looks clean, say so in one line.
- Output 3–7 bullets, severity-tagged (HIGH / MEDIUM / LOW).

Session ID for tool calls (may be empty if no repo was pre-ingested): {session_id}
"""

PR_SYNTHESIZER_PROMPT = """\
You are the synthesizer for a multi-agent PR review system.

Combine the three specialist outputs below into ONE markdown PR review with \
the following sections (use these exact headings):

## Summary
What the PR does, in 2–4 sentences.

## Impact
Affected modules / callers / tests, bulleted.

## Risks
Bulleted, severity-tagged. If nothing found, say "No significant risks identified."

## Suggested questions for the author
2–4 specific questions worth asking before merge.

Rules:
- Do not mention the specialist agents by name.
- Do not repeat the same fact in multiple sections.
- Be concrete: cite files, functions, lines.

## Specialist findings
### code_explainer
{code_explainer_output}

### impact_analyst
{impact_analyst_output}

### risk_assessor
{risk_assessor_output}

## PR metadata
Title: {pr_title}
Author: {pr_author}

Write the final review now.
"""
