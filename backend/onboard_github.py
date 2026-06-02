import re
import asyncio
import httpx
from config import settings
from models import RepoDetails
from utils.http_cache import cached_get

HEADERS = {
    "Authorization": f"Bearer {settings.github_token}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
BASE = "https://api.github.com"

MAX_FILE_SIZE = 20_000

PRIORITY_FILES = [
    "package.json", "composer.json", "pyproject.toml", "requirements.txt", "setup.py",
    "Cargo.toml", "go.mod", "Gemfile", "pom.xml", "build.gradle", "main.py", "app.py",
    "index.js", "index.ts", "app.js", "app.ts", "main.js", "main.ts", "main.go", "main.rs",
    "lib.rs", "next.config.js", "next.config.mjs", "vite.config.ts", "vite.config.js",
    "tsconfig.json", "docker-compose.yml", "Dockerfile"
]

PRIORITY_ORDER = [
    "package.json", "pyproject.toml", "cargo.toml", "go.mod", "requirements.txt",
    "main.py", "app.py", "index.ts", "index.js", "app.ts", "app.js", "main.ts", "main.js", "main.go", "main.rs",
    "docker-compose.yml", "dockerfile", "makefile", "tsconfig.json"
]

_CONTRIBUTING_CANDIDATES = [
    "CONTRIBUTING.md",
    "CONTRIBUTING.rst",
    "CONTRIBUTING.txt",
    ".github/CONTRIBUTING.md",
    "docs/CONTRIBUTING.md",
    "DEVELOPMENT.md",
    "docs/development.md",
    "HACKING.md",
]


async def _get(client: httpx.AsyncClient, url: str, params: dict = {}, ttl: float = 3600.0) -> dict | list:
    try:
        res = await cached_get(client, url, headers=HEADERS, params=params, ttl=ttl)
        res.raise_for_status()
        return res.json()
    except Exception:
        return {}


async def fetch_raw_file_content(client: httpx.AsyncClient, repo: str, filepath: str, max_chars: int = 20_000) -> str:
    headers = {**HEADERS, "Accept": "application/vnd.github.v3.raw"}
    try:
        res = await cached_get(client, f"{BASE}/repos/{repo}/contents/{filepath}", headers=headers, ttl=86400.0)
        res.raise_for_status()
        return res.text[:max_chars]
    except Exception:
        return ""


def should_ignore_path(path: str) -> bool:
    segments = path.split("/")
    ignored_dirs = {
        ".git", "node_modules", "venv", ".venv", "env", ".cache", "dist", "build",
        ".next", "__pycache__", ".idea", ".vscode", "target", "out"
    }
    for seg in segments:
        if seg in ignored_dirs:
            return True
            
    ignored_exts = {
        ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".zip", ".tar", ".gz", ".pdf",
        ".woff", ".woff2", ".eot", ".ttf", ".mp4", ".mp3", ".wav", ".exe", ".bin",
        ".pyc", ".db", ".sqlite", ".db3", ".sqlite3", ".o", ".a", ".so", ".dylib",
        ".lock", "-lock.json"
    }
    path_lower = path.lower()
    for ext in ignored_exts:
        if path_lower.endswith(ext):
            return True
    return False


def get_priority_weight(path: str) -> tuple[int, int]:
    basename = path.split("/")[-1].lower()
    depth = len(path.split("/"))
    weight = PRIORITY_ORDER.index(basename) if basename in PRIORITY_ORDER else len(PRIORITY_ORDER)
    return (weight, depth)


def build_tree(items: list[dict]) -> dict:
    root = {}
    for item in items:
        path = item.get("path", "")
        if not path or should_ignore_path(path):
            continue
        is_dir = (item.get("type") == "tree")
        parts = path.split("/")
        
        curr = root
        for i, part in enumerate(parts):
            is_last = (i == len(parts) - 1)
            if part not in curr:
                curr[part] = {
                    "is_dir": is_dir if is_last else True,
                    "children": {}
                }
            curr = curr[part]["children"]
    return root


def count_items_recursively(node: dict) -> int:
    count = 0
    for name, child in node["children"].items():
        count += 1
        if child["is_dir"]:
            count += count_items_recursively(child)
    return count


def format_tree(tree: dict, indent: str = "", depth: int = 1, max_depth: int = 3) -> list[str]:
    lines = []
    sorted_items = sorted(tree.items(), key=lambda x: (not x[1]["is_dir"], x[0].lower()))
    for name, node in sorted_items:
        prefix = "📁 " if node["is_dir"] else "📄 "
        if node["is_dir"]:
            if depth >= max_depth and node["children"]:
                hidden_count = count_items_recursively(node)
                lines.append(f"{indent}{prefix}{name}/... [{hidden_count} files/folders hidden]")
            else:
                lines.append(f"{indent}{prefix}{name}/")
                sub_lines = format_tree(node["children"], indent + "  ", depth + 1, max_depth)
                lines.extend(sub_lines)
        else:
            lines.append(f"{indent}{prefix}{name}")
    return lines


def find_readme_and_contributing_paths(tree_items: list[dict]) -> tuple[str | None, str | None]:
    readme_path = None
    contributing_path = None
    
    readme_candidates = ["readme.md", "readme.rst", "readme.txt", "readme"]
    contributing_candidates = [c.lower() for c in _CONTRIBUTING_CANDIDATES]
    
    sorted_items = sorted(tree_items, key=lambda x: len(x.get("path", "").split("/")))
    for item in sorted_items:
        if item.get("type") != "blob":
            continue
        path = item.get("path", "")
        if not path:
            continue
        path_lower = path.lower()
        basename = path.split("/")[-1].lower()
        
        if not readme_path and basename in readme_candidates:
            readme_path = path
        if not contributing_path and (basename in contributing_candidates or path_lower in contributing_candidates):
            contributing_path = path
            
    return readme_path, contributing_path


async def fetch_repo_info(
    client: httpx.AsyncClient, repo: str
) -> tuple[int, str, str, str, list[str], str]:
    data = await _get(client, f"{BASE}/repos/{repo}", ttl=3600.0)
    if not isinstance(data, dict):
        return 0, "", "", "", [], "main"
    stars = data.get("stargazers_count", 0)
    language = data.get("language") or ""
    license_name = (data.get("license") or {}).get("name", "")
    description = data.get("description") or ""
    topics = data.get("topics") or []
    default_branch = data.get("default_branch") or "main"
    return stars, language, license_name, description, topics, default_branch


def parse_github_repo_url(url: str) -> str:
    url = url.strip().rstrip("/")
    pattern = r"https?://github\.com/([^/]+/[^/]+)"
    match = re.match(pattern, url, re.IGNORECASE)
    if not match:
        raise ValueError("Invalid GitHub repository URL format. Must be like https://github.com/owner/repo")
    return match.group(1)


def parse_github_pr_url(url: str) -> tuple[str, str, int]:
    """Parse a GitHub PR URL into (owner, repo, pr_number).

    Accepts forms like https://github.com/owner/repo/pull/123 (with or without trailing slash).
    """
    url = url.strip().rstrip("/")
    pattern = r"https?://github\.com/([^/]+)/([^/]+)/pull/(\d+)"
    match = re.match(pattern, url, re.IGNORECASE)
    if not match:
        raise ValueError(
            "Invalid GitHub PR URL format. Must be like https://github.com/owner/repo/pull/123"
        )
    return match.group(1), match.group(2), int(match.group(3))


async def fetch_repo_details(repo: str) -> RepoDetails:
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: Fetch repository base details (including default branch)
        stars, language, license_name, description, topics, default_branch = await fetch_repo_info(client, repo)
        
        # Step 2: Fetch recursive git tree
        tree_url = f"{BASE}/repos/{repo}/git/trees/{default_branch}?recursive=true"
        tree_data = await _get(client, tree_url, ttl=3600.0)
        tree_items = tree_data.get("tree", []) if isinstance(tree_data, dict) else []
        
        # Step 3: Build and format directory structure
        tree_struct = build_tree(tree_items)
        structure = format_tree(tree_struct)
        
        # Step 4: Identify readme and contributing paths
        readme_path, contributing_path = find_readme_and_contributing_paths(tree_items)
        
        # Step 5: Gather priority paths
        priority_paths = []
        priority_basenames = {f.lower() for f in PRIORITY_FILES}
        deprioritized_dirs = {"tests", "test", "docs", "docs_src", "examples", "example", "website"}
        for item in tree_items:
            if item.get("type") != "blob":
                continue
            path = item.get("path", "")
            if not path or path in (readme_path, contributing_path) or should_ignore_path(path):
                continue
            basename = path.split("/")[-1].lower()
            if basename in priority_basenames:
                # Do not treat examples, tests, or documentation files as priority entry points
                path_segments = {s.lower() for s in path.split("/")}
                if not path_segments.intersection(deprioritized_dirs):
                    priority_paths.append(path)
                
        # Sort priority files by weight and depth
        priority_paths = sorted(priority_paths, key=get_priority_weight)
        
        # Fetch top 15 priority files concurrently
        priority_paths_to_fetch = priority_paths[:15]
        
        tasks = []
        if readme_path:
            tasks.append(("readme", fetch_raw_file_content(client, repo, readme_path, max_chars=20_000)))
        else:
            tasks.append(("readme", asyncio.sleep(0, result="")))
            
        if contributing_path:
            tasks.append(("contributing", fetch_raw_file_content(client, repo, contributing_path, max_chars=20_000)))
        else:
            tasks.append(("contributing", asyncio.sleep(0, result="")))
            
        for path in priority_paths_to_fetch:
            tasks.append((path, fetch_raw_file_content(client, repo, path, max_chars=8000)))
            
        # Execute concurrent fetching
        results = await asyncio.gather(*(t[1] for t in tasks))
        
        # Parse result contents
        readme = ""
        contributing = ""
        priority_contents = {}
        
        for (name, _), content in zip(tasks, results):
            if name == "readme":
                readme = content
            elif name == "contributing":
                contributing = content
            else:
                if content.strip():
                    priority_contents[name] = content
                    
        # Apply strict 15,000 character capping for combined priority files
        files = {}
        combined_chars = 0
        char_limit = 15_000
        
        for path in priority_paths_to_fetch:
            content = priority_contents.get(path, "")
            if not content:
                continue
            if combined_chars >= char_limit:
                break
            remaining = char_limit - combined_chars
            if len(content) > remaining:
                content = content[:remaining]
            files[path] = content
            combined_chars += len(content)
            
    return RepoDetails(
        repo=repo,
        readme=readme,
        contributing=contributing,
        repo_stars=stars,
        repo_language=language,
        license=license_name,
        description=description,
        topics=topics,
        files=files,
        structure=structure,
    )


async def fetch_specific_file(repo: str, filepath: str) -> str | None:
    async with httpx.AsyncClient(timeout=30.0) as client:
        content = await fetch_raw_file_content(client, repo, filepath, max_chars=6000)
        return content if content else None


