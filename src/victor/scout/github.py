"""A small, honest GitHub REST client.

Only the endpoints Scout needs, and one rule throughout: **the rate limit is a
budget, not an error to discover.** Unauthenticated GitHub allows 60 requests an
hour, which a portfolio analysis can exhaust in a single run - and the way that
usually presents is a report that silently analysed four repositories instead of
thirty. So the client counts what it spends, reports what is left, and refuses
loudly rather than returning a thinner answer that looks complete.

That is the same discipline as :mod:`victor.quota`, for the same reason. A free
tier you do not measure is one you will exceed halfway through something.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..errors import VictorError

API = "https://api.github.com"
USER_AGENT = "victor-agent/scout"

#: Beyond this, a README is boilerplate rather than signal.
README_CHARS = 4_000


class GitHubError(VictorError):
    """A GitHub request failed in a way worth stopping for."""

    exit_code = 7


class RateLimited(GitHubError):
    """The hourly budget is spent."""

    def __init__(self, resets_in: float, authenticated: bool) -> None:
        self.resets_in = resets_in
        minutes = max(1, int(resets_in // 60))
        hint = (
            "Set GITHUB_TOKEN for 5,000 requests an hour instead of 60."
            if not authenticated
            else "Wait for the window to roll over."
        )
        super().__init__(f"GitHub rate limit reached; resets in ~{minutes} min. {hint}")


@dataclass(frozen=True, slots=True)
class Repo:
    """One repository, reduced to what a portfolio comparison needs."""

    full_name: str
    name: str
    description: str = ""
    language: str = ""
    topics: tuple[str, ...] = ()
    stars: int = 0
    forks: int = 0
    pushed_at: str = ""
    url: str = ""
    readme: str = ""
    is_fork: bool = False
    archived: bool = False

    @property
    def owner(self) -> str:
        return self.full_name.split("/", 1)[0]

    def profile(self) -> str:
        """The text that represents this repo to the embedder.

        Ordered most-signal-first because the embedder truncates long input:
        topics and language say what a project *is*, the description says what
        it does, and the README is supporting detail that is often mostly
        badges and install instructions.
        """
        parts = [self.name.replace("-", " ").replace("_", " ")]
        if self.language:
            parts.append(f"written in {self.language}")
        if self.topics:
            parts.append("topics: " + ", ".join(self.topics))
        if self.description:
            parts.append(self.description)
        if self.readme:
            parts.append(self.readme[:README_CHARS])
        return "\n".join(parts)

    def __str__(self) -> str:
        return f"{self.full_name} ({self.stars}★)"


def _repo_from_json(raw: dict[str, Any], readme: str = "") -> Repo:
    return Repo(
        full_name=str(raw.get("full_name", "")),
        name=str(raw.get("name", "")),
        description=str(raw.get("description") or ""),
        language=str(raw.get("language") or ""),
        topics=tuple(raw.get("topics") or ()),
        stars=int(raw.get("stargazers_count") or 0),
        forks=int(raw.get("forks_count") or 0),
        pushed_at=str(raw.get("pushed_at") or ""),
        url=str(raw.get("html_url") or ""),
        readme=readme,
        is_fork=bool(raw.get("fork")),
        archived=bool(raw.get("archived")),
    )


@dataclass
class Budget:
    """What this run has spent, and what GitHub says is left."""

    spent: int = 0
    remaining: int | None = None
    limit: int | None = None
    reset_at: float = 0.0
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        if self.remaining is None:
            return f"{self.spent} requests"
        return f"{self.spent} requests, {self.remaining}/{self.limit} left this hour"


class GitHubClient:
    """Read-only access to the handful of endpoints Scout uses."""

    def __init__(
        self,
        token: str | None = None,
        *,
        client: httpx.Client | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.token = token
        self.budget = Budget()
        self._client = client or httpx.Client(timeout=timeout)

    @property
    def authenticated(self) -> bool:
        return bool(self.token)

    # -- transport ---------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": USER_AGENT,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _get(self, path: str, **params: Any) -> Any:
        url = path if path.startswith("http") else f"{API}{path}"
        try:
            response = self._client.get(url, headers=self._headers(), params=params or None)
        except httpx.HTTPError as exc:
            raise GitHubError(f"{type(exc).__name__}: {exc}") from exc

        self.budget.spent += 1
        self._record_limits(response)

        if response.status_code == 403 and self._is_rate_limit(response):
            raise RateLimited(
                max(0.0, self.budget.reset_at - time.time()), self.authenticated
            )
        if response.status_code == 401:
            raise GitHubError("GITHUB_TOKEN was rejected (HTTP 401)")
        if response.status_code == 404:
            raise GitHubError(f"not found: {url.removeprefix(API)}")
        if response.status_code >= 400:
            raise GitHubError(f"HTTP {response.status_code} for {url.removeprefix(API)}")
        return response.json()

    def _record_limits(self, response: httpx.Response) -> None:
        headers = response.headers
        try:
            if "x-ratelimit-remaining" in headers:
                self.budget.remaining = int(headers["x-ratelimit-remaining"])
            if "x-ratelimit-limit" in headers:
                self.budget.limit = int(headers["x-ratelimit-limit"])
            if "x-ratelimit-reset" in headers:
                self.budget.reset_at = float(headers["x-ratelimit-reset"])
        except (TypeError, ValueError):
            pass

    @staticmethod
    def _is_rate_limit(response: httpx.Response) -> bool:
        if response.headers.get("x-ratelimit-remaining") == "0":
            return True
        return "rate limit" in response.text.lower()

    def can_spend(self, requests: int) -> bool:
        """Whether ``requests`` more calls fit in what GitHub says is left."""
        return self.budget.remaining is None or self.budget.remaining >= requests

    # -- endpoints ---------------------------------------------------------

    def user_repos(self, user: str, *, limit: int = 60, include_forks: bool = False) -> list[Repo]:
        """Public repositories owned by ``user``, newest push first.

        Forks and archived repositories are excluded by default: a fork is
        somebody else's work and an archive is a statement that the work is
        over. Neither says much about what its owner can currently do, which
        is the question Scout is answering.
        """
        repos: list[Repo] = []
        page = 1
        while len(repos) < limit:
            batch = self._get(
                f"/users/{user}/repos", per_page=min(100, limit), page=page, sort="pushed"
            )
            if not isinstance(batch, list) or not batch:
                break
            for raw in batch:
                repo = _repo_from_json(raw)
                if repo.archived or (repo.is_fork and not include_forks):
                    continue
                repos.append(repo)
            if len(batch) < min(100, limit):
                break
            page += 1
        return repos[:limit]

    def readme(self, full_name: str) -> str:
        """A repository's README as text, or empty if it has none."""
        import base64

        try:
            payload = self._get(f"/repos/{full_name}/readme")
        except GitHubError:
            return ""
        content = payload.get("content") if isinstance(payload, dict) else None
        if not content:
            return ""
        try:
            return base64.b64decode(content).decode("utf-8", errors="replace")[:README_CHARS]
        except (ValueError, TypeError):
            return ""

    def search_repos(self, query: str, *, limit: int = 30) -> list[Repo]:
        """Repository search. See :mod:`victor.scout.corpus` on what this is not."""
        payload = self._get(
            "/search/repositories", q=query, sort="stars", order="desc", per_page=min(100, limit)
        )
        items = payload.get("items", []) if isinstance(payload, dict) else []
        return [_repo_from_json(raw) for raw in items[:limit]]

    def rate_limit(self) -> tuple[int, int]:
        """``(remaining, limit)`` without spending part of the budget on it.

        The ``/rate_limit`` endpoint is documented as not counting against the
        limit, which makes it the one thing worth calling before deciding
        whether a run can afford to start.
        """
        payload = self._get("/rate_limit")
        core = payload.get("resources", {}).get("core", {}) if isinstance(payload, dict) else {}
        self.budget.spent -= 1  # this call is free; do not report it as spent
        return int(core.get("remaining", 0)), int(core.get("limit", 0))

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GitHubClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
