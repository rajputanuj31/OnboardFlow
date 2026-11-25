"""GitHub: strict repo-URL parsing and public-repo metadata.

Security rules (design plan section 11):
- Only `https://github.com/<owner>/<repo>` is accepted. No other host, scheme, port,
  credentials, query or path, which blocks SSRF and `file://`-style tricks at the door.
- Public repositories only. The optional server token is used for metadata lookups (rate
  limit), never to read a private repo on behalf of a user: a repo the API reports as
  private is rejected even if the token could read it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

from onboardflow.settings import GitHubSettings

_URL_RE = re.compile(
    r"^https://github\.com/"
    r"(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/"
    r"(?P<name>[A-Za-z0-9._-]{1,100}?)"
    r"(?:\.git)?/?$"
)
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class InvalidRepoURLError(ValueError):
    pass


class RepoNotFoundError(LookupError):
    """Missing, or not public. Deliberately one error: never reveal private repo existence."""


class GitHubUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class RepoRef:
    owner: str
    name: str

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def clone_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}.git"


@dataclass(frozen=True)
class RepoInfo:
    ref: RepoRef  # canonical casing as reported by GitHub
    default_branch: str
    size_kb: int
    head_sha: str


def parse_repo_url(url: str) -> RepoRef:
    match = _URL_RE.match(url.strip())
    if not match or match["name"] in {".", ".."}:
        raise InvalidRepoURLError(
            "Only public GitHub repository URLs like https://github.com/owner/repo are supported."
        )
    return RepoRef(owner=match["owner"], name=match["name"])


class GitHubClient:
    def __init__(self, http: httpx.AsyncClient, settings: GitHubSettings) -> None:
        self.http = http
        self.base = str(settings.api_url).rstrip("/")
        self.headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "onboardflow",
        }
        if settings.token is not None:
            self.headers["Authorization"] = f"Bearer {settings.token.get_secret_value()}"
        self.timeout = settings.timeout_s

    async def _get(self, path: str, accept: str | None = None) -> httpx.Response:
        headers = {**self.headers, **({"Accept": accept} if accept else {})}
        try:
            response = await self.http.get(
                f"{self.base}{path}", headers=headers, timeout=self.timeout
            )
        except httpx.HTTPError as exc:
            raise GitHubUnavailableError(f"GitHub API unreachable: {type(exc).__name__}") from exc
        if response.status_code == 404:
            raise RepoNotFoundError("Repository not found or not public.")
        if response.status_code >= 400:
            raise GitHubUnavailableError(f"GitHub API returned {response.status_code}")
        return response

    async def repo_info(self, ref: RepoRef) -> RepoInfo:
        data = (await self._get(f"/repos/{ref.owner}/{ref.name}")).json()
        if data.get("private") or data.get("visibility", "public") != "public":
            raise RepoNotFoundError("Repository not found or not public.")
        canonical = RepoRef(owner=data["owner"]["login"], name=data["name"])
        branch = str(data["default_branch"])
        sha = (
            await self._get(
                f"/repos/{canonical.owner}/{canonical.name}/commits/{branch}",
                accept="application/vnd.github.sha",
            )
        ).text.strip()
        if not _SHA_RE.match(sha):
            raise GitHubUnavailableError("GitHub returned an invalid commit sha")
        return RepoInfo(
            ref=canonical, default_branch=branch, size_kb=int(data.get("size", 0)), head_sha=sha
        )
