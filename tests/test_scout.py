"""P7: the GitHub client, the corpus heuristic, and the gap ranking.

No network. The client is driven through ``httpx.MockTransport``, the same way
the provider layer has been tested since P0, so rate-limit handling and error
paths are exercised without spending any of a 60-an-hour budget on a test run.
"""

from __future__ import annotations

import base64

import httpx
import pytest

from victor.rag.embed import HashEmbedder
from victor.scout import GitHubClient, GitHubError, RateLimited, Repo, analyse
from victor.scout.corpus import (
    Corpus,
    Seed,
    build_corpus,
    build_query,
    seed_topics,
)

RATE_HEADERS = {
    "x-ratelimit-limit": "60",
    "x-ratelimit-remaining": "59",
    "x-ratelimit-reset": "9999999999",
}


def repo_json(full_name: str, **overrides) -> dict:
    owner, name = full_name.split("/")
    payload = {
        "full_name": full_name,
        "name": name,
        "description": f"{name} does something",
        "language": "Python",
        "topics": ["cli"],
        "stargazers_count": 100,
        "forks_count": 3,
        "pushed_at": "2026-01-01T00:00:00Z",
        "html_url": f"https://github.com/{full_name}",
        "fork": False,
        "archived": False,
    }
    payload.update(overrides)
    return payload


def client_for(handler, token: str | None = None) -> GitHubClient:
    transport = httpx.MockTransport(handler)
    return GitHubClient(token, client=httpx.Client(transport=transport))


# --- the client ------------------------------------------------------------


def test_forks_and_archives_are_excluded():
    """A fork is somebody else's work; an archive says the work is over."""

    def handler(request):
        return httpx.Response(
            200,
            json=[
                repo_json("u/real"),
                repo_json("u/forked", fork=True),
                repo_json("u/old", archived=True),
            ],
            headers=RATE_HEADERS,
        )

    repos = client_for(handler).user_repos("u")
    assert [r.name for r in repos] == ["real"]


def test_forks_can_be_asked_for():
    def handler(request):
        return httpx.Response(
            200, json=[repo_json("u/a"), repo_json("u/b", fork=True)], headers=RATE_HEADERS
        )

    repos = client_for(handler).user_repos("u", include_forks=True)
    assert len(repos) == 2


def test_the_rate_limit_is_reported_not_discovered():
    """A budget you do not measure is one you exceed halfway through a run."""

    def handler(request):
        return httpx.Response(200, json=[repo_json("u/a")], headers=RATE_HEADERS)

    client = client_for(handler)
    client.user_repos("u")
    assert client.budget.spent == 1
    assert client.budget.remaining == 59
    assert "59/60 left" in client.budget.describe()


def test_exhausting_the_budget_raises_with_a_way_out():
    def handler(request):
        return httpx.Response(
            403,
            json={"message": "API rate limit exceeded"},
            headers={**RATE_HEADERS, "x-ratelimit-remaining": "0"},
        )

    with pytest.raises(RateLimited, match="GITHUB_TOKEN"):
        client_for(handler).user_repos("u")


def test_an_authenticated_client_is_told_to_wait_instead():
    def handler(request):
        return httpx.Response(
            403, json={}, headers={**RATE_HEADERS, "x-ratelimit-remaining": "0"}
        )

    with pytest.raises(RateLimited, match="Wait for the window"):
        client_for(handler, token="t").user_repos("u")


def test_can_spend_reflects_what_github_said():
    def handler(request):
        return httpx.Response(
            200, json=[repo_json("u/a")], headers={**RATE_HEADERS, "x-ratelimit-remaining": "3"}
        )

    client = client_for(handler)
    client.user_repos("u")
    assert client.can_spend(3)
    assert not client.can_spend(4)


def test_a_rejected_token_says_so():
    def handler(request):
        return httpx.Response(401, json={})

    with pytest.raises(GitHubError, match="401"):
        client_for(handler, token="bad").user_repos("u")


def test_a_missing_user_is_a_clear_error():
    def handler(request):
        return httpx.Response(404, json={})

    with pytest.raises(GitHubError, match="not found"):
        client_for(handler).user_repos("nobody")


def test_a_readme_is_decoded():
    def handler(request):
        content = base64.b64encode(b"# Title\nsome prose").decode()
        return httpx.Response(200, json={"content": content}, headers=RATE_HEADERS)

    assert "some prose" in client_for(handler).readme("u/r")


def test_a_missing_readme_is_empty_not_an_error():
    """Plenty of repositories have none, and that is not a failure."""

    def handler(request):
        return httpx.Response(404, json={})

    assert client_for(handler).readme("u/r") == ""


def test_the_rate_limit_probe_does_not_count_against_itself():
    def handler(request):
        return httpx.Response(
            200,
            json={"resources": {"core": {"remaining": 42, "limit": 60}}},
            headers=RATE_HEADERS,
        )

    client = client_for(handler)
    assert client.rate_limit() == (42, 60)
    assert client.budget.spent == 0


# --- the corpus ------------------------------------------------------------


def make_repo(name: str, topics=(), language="Python", stars=0, description="") -> Repo:
    return Repo(
        full_name=f"o/{name}",
        name=name,
        topics=tuple(topics),
        language=language,
        stars=stars,
        description=description,
    )


def test_generic_topics_are_not_worth_searching():
    """A corpus seeded on "python" is a corpus of everything."""
    seeds = seed_topics([make_repo("a", topics=["python", "cli", "hacktoberfest"])])
    assert [s.value for s in seeds if s.kind == "topic"] == ["cli"]


def test_languages_fill_in_when_there_are_no_topics():
    """Found by running it: a portfolio with no topics produced no corpus.

    Languages were being filtered by the same generic-topic list, so a real
    account with ten untagged repositories got an empty comparison set and a
    report saying nothing was wrong.
    """
    seeds = seed_topics([make_repo("a", language="Python"), make_repo("b", language="C")])
    assert {s.value for s in seeds} == {"Python", "C"}
    assert all(s.kind == "language" for s in seeds)


def test_topics_are_preferred_over_languages():
    seeds = seed_topics(
        [make_repo("a", topics=["raytracing"], language="C")], limit=2
    )
    assert seeds[0] == Seed("raytracing", "topic")


def test_a_language_query_is_not_a_topic_query():
    assert build_query(Seed("cli", "topic")).startswith("topic:cli ")
    assert build_query(Seed("Python", "language")).startswith("language:Python ")
    assert "stars:>200" in build_query(Seed("cli", "topic"))


def test_the_corpus_excludes_the_users_own_repositories():
    """Your own work cannot be evidence of a gap in your own portfolio."""
    mine = [make_repo("mine", topics=["cli"])]

    class FakeClient:
        budget = None

        def can_spend(self, n):
            return True

        def search_repos(self, query, limit):
            return [make_repo("mine", topics=["cli"]), make_repo("theirs", topics=["cli"])]

    corpus = build_corpus(FakeClient(), mine)
    assert [r.name for r in corpus.repos] == ["theirs"]


def test_the_corpus_stops_early_rather_than_failing():
    """Four topics with the other two named beats an exception."""
    mine = [make_repo("a", topics=["one", "two", "three"])]
    calls = []

    class Stingy:
        def can_spend(self, n):
            return len(calls) < 1

        def search_repos(self, query, limit):
            calls.append(query)
            return [make_repo("x", topics=["one"])]

    corpus = build_corpus(Stingy(), mine)
    assert len(corpus.repos) == 1
    assert any("stopped early" in note for note in corpus.query_notes)


def test_a_failed_search_does_not_end_the_run():
    mine = [make_repo("a", topics=["one", "two"])]

    class Flaky:
        def can_spend(self, n):
            return True

        def search_repos(self, query, limit):
            if "one" in query:
                raise RuntimeError("boom")
            return [make_repo("x", topics=["two"])]

    corpus = build_corpus(Flaky(), mine)
    assert len(corpus.repos) == 1
    assert any("search failed" in note for note in corpus.query_notes)


def test_provenance_says_what_this_is_and_is_not():
    """The person reading a ranked list needs to know how it was made."""
    corpus = Corpus((make_repo("x"),), ("topic:cli",), ())
    assert "no trending API" in corpus.provenance
    assert "heuristic" in corpus.provenance


# --- the ranking -----------------------------------------------------------


def analysis(user_repos, corpus_repos, **kwargs):
    corpus = Corpus(tuple(corpus_repos), ("topic:x",), ())
    return analyse(HashEmbedder(), "u", user_repos, corpus, **kwargs)


def test_every_gap_cites_the_repositories_that_produced_it():
    """The exit gate's requirement, and the difference from a horoscope."""
    mine = [make_repo("notes", topics=["markdown"], description="a notes app")]
    theirs = [
        make_repo("k8s-a", topics=["kubernetes"], stars=900, description="cluster orchestration"),
        make_repo("k8s-b", topics=["kubernetes"], stars=400, description="container scheduling"),
    ]
    report = analysis(mine, theirs)

    assert report.gaps, "expected kubernetes to rank as a gap"
    gap = report.gaps[0]
    assert gap.topic == "kubernetes"
    assert "o/k8s-a" in gap.cite()
    assert gap.nearest_repo == "o/notes"
    assert gap.prevalence == 2


def test_evidence_is_ordered_by_stars():
    mine = [make_repo("notes", topics=["markdown"])]
    theirs = [
        make_repo("small", topics=["kubernetes"], stars=10),
        make_repo("big", topics=["kubernetes"], stars=5000),
    ]
    assert analysis(mine, theirs).gaps[0].cite().startswith("o/big")


def test_a_topic_held_by_one_repository_is_not_a_field():
    mine = [make_repo("notes", topics=["markdown"])]
    theirs = [make_repo("lonely", topics=["esoteric"], stars=10)] + [
        make_repo(f"k{i}", topics=["kubernetes"], stars=10) for i in range(3)
    ]
    assert "esoteric" not in {gap.topic for gap in analysis(mine, theirs).gaps}


def test_near_duplicate_topics_collapse_into_one_row():
    """`style-guide` and `styleguide` were two identical rows in a real run."""
    mine = [make_repo("notes", topics=["markdown"])]
    theirs = [
        make_repo("a", topics=["style-guide", "styleguide"], stars=10),
        make_repo("b", topics=["styleguide"], stars=20),
    ]
    topics = [g.topic for g in analysis(mine, theirs).gaps]
    assert topics.count("styleguide") + topics.count("style-guide") <= 1


def test_distance_is_reported_as_a_rank_not_a_score():
    """Absolute similarities all land in one narrow band; ranks do not."""
    mine = [make_repo("notes", topics=["markdown"])]
    theirs = [make_repo(f"k{i}", topics=["kubernetes"], stars=10) for i in range(3)]
    gap = analysis(mine, theirs).gaps[0]

    assert 0.0 <= gap.percentile <= 1.0
    assert "% of the set" in gap.distance_note


def test_an_empty_corpus_produces_an_empty_report():
    report = analysis([make_repo("a")], [])
    assert not report.found
    assert report.repos_analysed == 1


def test_a_user_with_no_repositories_is_reported_not_crashed():
    report = analysis([], [make_repo("x", topics=["kubernetes"])])
    assert not report.found
    assert "could not find any public repositories" in report.spoken()


# --- the spoken summary ----------------------------------------------------


def test_the_spoken_summary_is_short_sentences():
    """Piper streams a chunk per sentence, so several short ones start sooner."""
    mine = [make_repo("notes", topics=["markdown"])]
    theirs = [make_repo(f"k{i}", topics=["kubernetes"], stars=10) for i in range(3)]
    spoken = analysis(mine, theirs).spoken()

    sentences = [s for s in spoken.split(".") if s.strip()]
    assert len(sentences) >= 3
    assert max(len(s) for s in sentences) < 120


def test_the_spoken_summary_carries_no_stars_or_decimals():
    """Hearing "zero point four two" is not information."""
    mine = [make_repo("notes", topics=["markdown"])]
    theirs = [make_repo(f"k{i}", topics=["kubernetes"], stars=54321) for i in range(3)]
    spoken = analysis(mine, theirs).spoken()

    assert "54321" not in spoken and "★" not in spoken
    assert "0." not in spoken


def test_the_spoken_summary_admits_what_it_is():
    mine = [make_repo("notes", topics=["markdown"])]
    theirs = [make_repo(f"k{i}", topics=["kubernetes"], stars=10) for i in range(3)]
    assert "heuristic" in analysis(mine, theirs).spoken()


# --- the repo profile ------------------------------------------------------


def test_the_profile_leads_with_what_a_project_is():
    repo = Repo(
        full_name="o/ray-tracer",
        name="ray-tracer",
        language="Rust",
        topics=("graphics", "raytracing"),
        description="a path tracer",
        readme="# install\npip install x" * 500,
    )
    profile = repo.profile()

    assert profile.index("Rust") < profile.index("a path tracer")
    assert profile.index("graphics") < profile.index("a path tracer")
    # Hyphens become spaces so the name reads as words to an embedder.
    assert profile.startswith("ray tracer")
    assert len(profile) < 6000


def test_json_shapes_that_github_actually_returns():
    """Nulls, not missing keys, are how GitHub says "no language"."""

    def handler(request):
        return httpx.Response(
            200,
            json=[repo_json("u/a", language=None, description=None, topics=None)],
            headers=RATE_HEADERS,
        )

    repo = client_for(handler).user_repos("u")[0]
    assert repo.language == "" and repo.description == "" and repo.topics == ()


def test_search_results_are_parsed_from_the_items_envelope():
    def handler(request):
        return httpx.Response(
            200, json={"items": [repo_json("o/x", stargazers_count=900)]}, headers=RATE_HEADERS
        )

    found = client_for(handler).search_repos("topic:cli stars:>200")
    assert found[0].stars == 900


def test_the_search_query_reaches_github_intact():
    seen = {}

    def handler(request):
        seen["q"] = dict(httpx.QueryParams(request.url.query.decode()))["q"]
        return httpx.Response(200, json={"items": []}, headers=RATE_HEADERS)

    client_for(handler).search_repos("topic:cli stars:>200 pushed:>2026-01-01")
    assert seen["q"] == "topic:cli stars:>200 pushed:>2026-01-01"


def test_the_budget_survives_json_without_rate_headers():
    def handler(request):
        return httpx.Response(200, json=[repo_json("u/a")])

    client = client_for(handler)
    client.user_repos("u")
    assert client.budget.remaining is None
    assert client.can_spend(100), "unknown budget must not block the run"
