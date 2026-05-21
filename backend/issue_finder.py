import httpx
import asyncio
from config import settings
from models import UserInput, MatchedIssue, IssueSearchResult
from query_builder import QueryBuilder
from issue_scorer import IssueScorer

HEADERS = {
    "Authorization": f"Bearer {settings.github_token}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28"
}

REPO_MIN_STARS = 500
REPO_MIN_GOOD_FIRST_ISSUES = 3


class IssueFinder:
    def __init__(self):
        self.query_builder = QueryBuilder()
        self.scorer = IssueScorer()

    async def _search_repos(
        self, client: httpx.AsyncClient, language: str = "", topic: str = ""
    ) -> list[str]:
        """Search GitHub for quality repos by language or topic."""
        query_parts = [
            f"good-first-issues:>={REPO_MIN_GOOD_FIRST_ISSUES}",
            f"stars:>={REPO_MIN_STARS}",
            "pushed:>2024-01-01",
        ]
        if language:
            query_parts.append(f"language:{language}")
        if topic:
            query_parts.append(f"topic:{topic}")

        try:
            res = await client.get(
                "https://api.github.com/search/repositories",
                headers=HEADERS,
                params={
                    "q": " ".join(query_parts),
                    "sort": "good-first-issues",
                    "order": "desc",
                    "per_page": 3,
                },
            )
            if res.status_code != 200:
                return []
            return [item["full_name"] for item in res.json().get("items", [])]
        except Exception:
            return []

    async def _fetch_issues(
        self, client: httpx.AsyncClient, repo: str
    ) -> list[MatchedIssue]:
        """Fetch open good-first-issue and help-wanted issues from a repo."""
        issues: list[MatchedIssue] = []
        for label in ["good first issue", "help wanted"]:
            try:
                res = await client.get(
                    f"https://api.github.com/repos/{repo}/issues",
                    headers=HEADERS,
                    params={"labels": label, "state": "open", "per_page": 3},
                )
                if res.status_code != 200:
                    continue
                for item in res.json():
                    if "pull_request" in item:
                        continue
                    issues.append(MatchedIssue(
                        repo=repo,
                        issue_number=item["number"],
                        title=item["title"],
                        url=item["html_url"],
                        labels=[l["name"] for l in item.get("labels", [])],
                        match_reason="",  # filled in by scorer
                    ))
            except Exception:
                continue
        return issues[:3]

    async def find(self, user_input: UserInput) -> IssueSearchResult:
        # Step 1: LLM translates stack + interests into GitHub search terms
        search_terms = await self.query_builder.build(
            user_input.stack, user_input.interests
        )

        async with httpx.AsyncClient(timeout=20.0) as client:
            # Step 2: Search repos by language AND by topic (parallel)
            repo_searches = [
                self._search_repos(client, language=lang)
                for lang in search_terms.languages
            ] + [
                self._search_repos(client, topic=topic)
                for topic in search_terms.topics
            ]

            repo_results = await asyncio.gather(*repo_searches, return_exceptions=True)

            # Deduplicate repos, cap at 8
            seen: set[str] = set()
            all_repos: list[str] = []
            for batch in repo_results:
                if not isinstance(batch, list):
                    continue
                for repo in batch:
                    if repo not in seen:
                        seen.add(repo)
                        all_repos.append(repo)
            all_repos = all_repos[:8]

            # Step 3: Fetch issues from all repos (parallel)
            issue_results = await asyncio.gather(*[
                self._fetch_issues(client, repo)
                for repo in all_repos
            ], return_exceptions=True)

        raw_issues = [
            issue
            for batch in issue_results
            if isinstance(batch, list)
            for issue in batch
        ]

        # Step 4: LLM scores and ranks issues by relevance
        ranked_issues = await self.scorer.score(
            raw_issues, user_input.stack, user_input.interests
        )

        return IssueSearchResult(
            issues=ranked_issues[:10],
            searched_repos=all_repos,
        )
