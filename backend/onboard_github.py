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


async def _get(client: httpx.AsyncClient, url: str, params: dict = {}, ttl: float = 3600.0) -> dict | list:
    try:
        res = await cached_get(client, url, headers=HEADERS, params=params, ttl=ttl)
        res.raise_for_status()
        return res.json()
    except Exception:
        return {}


async def _fetch_file_by_name(client: httpx.AsyncClient, repo: str, filename: str) -> str:
    data = await _get(client, f"{BASE}/repos/{repo}/contents/{filename}", ttl=86400.0)
    if not data or "download_url" not in data:
        return ""
    try:
        res = await cached_get(client, data["download_url"], ttl=86400.0)
        return res.text[:MAX_FILE_SIZE]
    except Exception:
        return ""


async def fetch_readme(client: httpx.AsyncClient, repo: str) -> str:
    data = await _get(client, f"{BASE}/repos/{repo}/readme", ttl=86400.0)
    if not data or "download_url" not in data:
        return ""
    try:
        res = await cached_get(client, data["download_url"], ttl=86400.0)
        return res.text[:MAX_FILE_SIZE]
    except Exception:
        return ""


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


async def fetch_contributing(client: httpx.AsyncClient, repo: str) -> str:
    for filename in _CONTRIBUTING_CANDIDATES:
        content = await _fetch_file_by_name(client, repo, filename)
        if content:
            return content
    return ""


async def fetch_repo_info(
    client: httpx.AsyncClient, repo: str
) -> tuple[int, str, str]:
    data = await _get(client, f"{BASE}/repos/{repo}", ttl=3600.0)
    if not isinstance(data, dict):
        return 0, "", ""
    stars = data.get("stargazers_count", 0)
    language = data.get("language") or ""
    license_name = (data.get("license") or {}).get("name", "")
    return stars, language, license_name


def parse_github_repo_url(url: str) -> str:
    url = url.strip().rstrip("/")
    pattern = r"https?://github\.com/([^/]+/[^/]+)"
    match = re.match(pattern, url, re.IGNORECASE)
    if not match:
        raise ValueError("Invalid GitHub repository URL format. Must be like https://github.com/owner/repo")
    return match.group(1)


async def fetch_repo_details(repo: str) -> RepoDetails:
    async with httpx.AsyncClient(timeout=30.0) as client:
        (stars, language, license_name), readme, contributing = (
            await asyncio.gather(
                fetch_repo_info(client, repo),
                fetch_readme(client, repo),
                fetch_contributing(client, repo),
            )
        )

    return RepoDetails(
        repo=repo,
        readme=readme,
        contributing=contributing,
        repo_stars=stars,
        repo_language=language,
        license=license_name,
    )
