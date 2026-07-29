"""P7: GitHub portfolio gap analysis.

A secondary feature, deliberately - the plan names it the first thing to cut if
anything overran. It ships because P6 did: the embedder and the similarity
maths are reused wholesale, so Scout is three files of GitHub plumbing rather
than a second product.

The output is a heuristic and says so on every run. GitHub has no trending API,
stars measure attention rather than quality, and a corpus seeded from the user's
own topics finds gaps adjacent to what they already do. Each row cites the
repositories that produced it, so a reader can disagree with something specific.
"""

from .analyze import Gap, Report, Strength, analyse, cosine
from .corpus import Corpus, build_corpus, build_query, seed_topics
from .github import Budget, GitHubClient, GitHubError, RateLimited, Repo

__all__ = [
    "Budget",
    "Corpus",
    "Gap",
    "GitHubClient",
    "GitHubError",
    "RateLimited",
    "Report",
    "Repo",
    "Strength",
    "analyse",
    "build_corpus",
    "build_query",
    "cosine",
    "seed_topics",
]


def scout(
    user: str,
    *,
    settings,
    limit: int = 8,
    max_repos: int = 30,
    readmes: int = 8,
    embedder=None,
    client: GitHubClient | None = None,
) -> Report:
    """Run the whole analysis for one GitHub user.

    ``readmes`` is a budget, not a preference: each one is a request, and the
    unauthenticated allowance is 60 an hour. The most recently pushed
    repositories get them, on the reasoning that the newest work is the most
    representative of what someone can currently do.
    """
    from dataclasses import replace

    from ..rag.embed import select_embedder
    from .analyze import analyse
    from .corpus import build_corpus

    owned = client or GitHubClient(settings.secret("github_token"))
    chosen = embedder or select_embedder(settings.paths.ensure().models_dir)

    repos = owned.user_repos(user, limit=max_repos)
    for index, repo in enumerate(repos[:readmes]):
        # Leave room for the corpus searches: a report with READMEs and no
        # comparison is not a report.
        if not owned.can_spend(len(repos[:readmes]) - index + 2):
            break
        text = owned.readme(repo.full_name)
        if text:
            repos[index] = replace(repo, readme=text)

    corpus = build_corpus(owned, repos)
    return analyse(
        chosen, user, repos, corpus, limit=limit, budget=owned.budget.describe()
    )
