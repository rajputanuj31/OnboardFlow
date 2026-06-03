"""parse_github_pr_url: happy path + rejection of non-PR URLs."""
import pytest

from onboard_github import parse_github_pr_url


def test_parse_basic():
    assert parse_github_pr_url("https://github.com/psf/requests/pull/100") == ("psf", "requests", 100)


def test_parse_trailing_slash():
    assert parse_github_pr_url("https://github.com/owner/repo/pull/42/") == ("owner", "repo", 42)


def test_parse_http_scheme():
    assert parse_github_pr_url("http://github.com/owner/repo/pull/7") == ("owner", "repo", 7)


def test_rejects_issue_url():
    with pytest.raises(ValueError):
        parse_github_pr_url("https://github.com/owner/repo/issues/5")


def test_rejects_bare_repo_url():
    with pytest.raises(ValueError):
        parse_github_pr_url("https://github.com/owner/repo")


def test_rejects_garbage():
    with pytest.raises(ValueError):
        parse_github_pr_url("not-a-url")
