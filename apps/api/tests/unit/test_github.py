from __future__ import annotations

import httpx
import pytest
from pydantic import SecretStr

from onboardflow.github import (
    GitHubClient,
    GitHubUnavailableError,
    InvalidRepoURLError,
    RepoNotFoundError,
    RepoRef,
    parse_repo_url,
)
from onboardflow.settings import GitHubSettings

SHA = "a" * 40


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://github.com/psf/requests", RepoRef("psf", "requests")),
        ("https://github.com/psf/requests/", RepoRef("psf", "requests")),
        ("https://github.com/psf/requests.git", RepoRef("psf", "requests")),
        ("  https://github.com/pallets/flask  ", RepoRef("pallets", "flask")),
        ("https://github.com/a-b/c.d_e", RepoRef("a-b", "c.d_e")),
    ],
)
def test_parse_accepts_plain_repo_urls(url: str, expected: RepoRef) -> None:
    assert parse_repo_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/psf/requests",  # not https
        "https://gitlab.com/psf/requests",  # other host
        "https://github.com.evil.com/psf/requests",
        "https://user:pw@github.com/psf/requests",  # credentials
        "https://github.com:8443/psf/requests",  # port
        "https://github.com/psf/requests/tree/main",  # extra path
        "https://github.com/psf/requests?x=1",
        "https://github.com/psf",
        "https://github.com/psf/..",
        "file:///etc/passwd",
        "git@github.com:psf/requests.git",
        "https://github.com/-bad/requests",
    ],
)
def test_parse_rejects_everything_else(url: str) -> None:
    with pytest.raises(InvalidRepoURLError):
        parse_repo_url(url)


def _client(handler: httpx.MockTransport, token: str | None = None) -> GitHubClient:
    settings = GitHubSettings(token=SecretStr(token) if token else None)
    return GitHubClient(httpx.AsyncClient(transport=handler), settings)


def _api(repo: dict[str, object] | None, sha: str = SHA, status: int = 200) -> httpx.MockTransport:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if status != 200:
            return httpx.Response(status)
        if request.url.path.endswith("/commits/main"):
            return httpx.Response(200, text=sha)
        return httpx.Response(200, json=repo)

    transport = httpx.MockTransport(handler)
    transport.seen = seen  # type: ignore[attr-defined]
    return transport


PUBLIC = {
    "owner": {"login": "PSF"},
    "name": "Requests",
    "default_branch": "main",
    "size": 9000,
    "private": False,
    "visibility": "public",
}


async def test_public_repo_resolves_canonical_name_branch_and_head() -> None:
    info = await _client(_api(PUBLIC)).repo_info(RepoRef("psf", "requests"))
    assert info.ref == RepoRef("PSF", "Requests")
    assert info.default_branch == "main"
    assert info.head_sha == SHA
    assert info.size_kb == 9000


async def test_private_repo_is_rejected_even_if_the_token_can_read_it() -> None:
    private = {**PUBLIC, "private": True, "visibility": "private"}
    with pytest.raises(RepoNotFoundError):
        await _client(_api(private), token="server-token").repo_info(RepoRef("o", "r"))


async def test_missing_repo_and_api_errors() -> None:
    with pytest.raises(RepoNotFoundError):
        await _client(_api(None, status=404)).repo_info(RepoRef("o", "r"))
    with pytest.raises(GitHubUnavailableError):
        await _client(_api(None, status=500)).repo_info(RepoRef("o", "r"))
    with pytest.raises(GitHubUnavailableError, match="invalid commit"):
        await _client(_api(PUBLIC, sha="<html>")).repo_info(RepoRef("o", "r"))


async def test_token_header_only_when_configured() -> None:
    anonymous = _api(PUBLIC)
    await _client(anonymous).repo_info(RepoRef("o", "r"))
    assert "authorization" not in anonymous.seen[0].headers  # type: ignore[attr-defined]

    with_token = _api(PUBLIC)
    await _client(with_token, token="t0k").repo_info(RepoRef("o", "r"))
    assert with_token.seen[0].headers["authorization"] == "Bearer t0k"  # type: ignore[attr-defined]
