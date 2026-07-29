"""Ranking the gaps, and showing the working.

The analysis is deliberately simple, because a portfolio comparison that cannot
be checked is worth nothing. Every repository - the user's and the corpus's -
becomes one vector using P6's embedder. For each topic in the corpus, coverage
is the closest any of the user's repositories gets to any repository carrying
that topic. Low coverage across a widely-held topic is a gap.

**Every row cites its evidence.** The plan is explicit about this and it is the
difference between advice and horoscope: a row says which corpus repositories
produced it, which of the user's repositories came closest, and the number that
separated them. A reader who disagrees can see exactly what to disagree with.

**The numbers are ranks, not scores.** Running this against a real account
showed the raw similarities spanning 0.53 to 0.73 across the whole comparison
set - a sentence embedder puts nearly all software writing in a narrow band, so
an absolute threshold decides nothing. Each row is therefore placed within
*this run's own* distribution, and the report phrases it that way: "further from
your work than 80% of the set" is a claim that survives inspection, while
"coverage 0.58" reads as a measurement and is not one.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from .corpus import GENERIC_TOPICS, Corpus
from .github import Repo

MIN_PREVALENCE = 2
"""A topic held by one repository is that repository's business, not a field."""

GAP_PERCENTILE = 0.45
STRENGTH_PERCENTILE = 0.75
"""Where a topic sits in *this run's own* distribution of similarities.

Absolute thresholds do not work here, and finding that out is what running it
against a real account was for. Measured across 48 corpus repositories: the
nearest-user similarity ranged from 0.53 to 0.73, with a median of 0.63. Every
value a sentence embedder produces for two pieces of software writing lands in
that band, so a fixed floor of 0.55 called everything "covered" and a floor of
0.65 would have called half of it a gap, arbitrarily.

Ranking within the run is robust to that compression, and it is also the more
honest claim: Scout can say "this is further from your work than most of the
comparison set", which is true, rather than "your coverage is 0.58", which
sounds like a measurement and is not."""


def cosine(a: list[float], b: list[float]) -> float:
    """Both vectors come from the embedder already normalised."""
    return sum(x * y for x, y in zip(a, b, strict=True))


@dataclass(frozen=True, slots=True)
class Gap:
    """One topic the portfolio does not reach, and the evidence for saying so."""

    topic: str
    coverage: float
    prevalence: int
    evidence: tuple[Repo, ...]
    nearest_repo: str = ""
    percentile: float = 0.0
    """Where this topic's coverage sits among all corpus repositories, 0-1.
    0.1 means "closer to your work than only 10% of the comparison set"."""

    @property
    def score(self) -> float:
        """How much of a gap this is.

        Rank distance, weighted by how established the topic is in the corpus -
        so a thing nobody does is not ranked above a thing everybody does.
        Prevalence is capped at five repositories, or one popular topic would
        dominate every report.
        """
        weight = min(self.prevalence, 5) / 5
        return (1.0 - self.percentile) * (0.5 + 0.5 * weight)

    @property
    def distance_note(self) -> str:
        """The row's number, phrased as the relative claim it actually is."""
        return f"further from your work than {(1 - self.percentile) * 100:.0f}% of the set"

    def cite(self, limit: int = 3) -> str:
        """The specific repositories behind this row."""
        return ", ".join(f"{r.full_name} ({r.stars}★)" for r in self.evidence[:limit])


@dataclass(frozen=True, slots=True)
class Strength:
    """A topic the portfolio already covers, named so the report is balanced."""

    topic: str
    coverage: float
    repo: str


@dataclass(frozen=True, slots=True)
class Report:
    """The ranked gap analysis, with everything needed to check it."""

    user: str
    gaps: tuple[Gap, ...]
    strengths: tuple[Strength, ...]
    corpus: Corpus
    repos_analysed: int
    budget: str = ""

    @property
    def found(self) -> bool:
        return bool(self.gaps)

    def spoken(self, limit: int = 3) -> str:
        """A few short sentences, for reading aloud.

        Short ones on purpose: Piper streams a chunk per sentence, so several
        short sentences start playing sooner than one long one. Star counts and
        decimals are dropped - they are for the printed table, and hearing
        "zero point four two" is not information.
        """
        if not self.repos_analysed:
            return f"I could not find any public repositories for {self.user}."
        if not self.gaps:
            return (
                f"I compared {self.repos_analysed} of your repositories against "
                f"{len(self.corpus)} others. Nothing stood out as a gap."
            )
        named = [gap.topic.replace("-", " ") for gap in self.gaps[:limit]]
        sentences = [
            f"I compared {self.repos_analysed} of your repositories "
            f"against {len(self.corpus)} active ones.",
            f"The biggest gap is {named[0]}.",
        ]
        if len(named) > 1:
            sentences.append("Then " + " and ".join(named[1:]) + ".")
        if self.strengths:
            sentences.append(f"You are already strong on {self.strengths[0].topic}.")
        sentences.append("This is a heuristic, not a ranking. Check the evidence.")
        return " ".join(sentences)


def _profile_vectors(embedder: Any, repos: list[Repo]) -> list[list[float]]:
    return embedder.encode([repo.profile() for repo in repos]) if repos else []


def analyse(
    embedder: Any,
    user: str,
    user_repos: list[Repo],
    corpus: Corpus,
    *,
    limit: int = 8,
    budget: str = "",
) -> Report:
    """Compare a portfolio against a corpus and rank what is missing."""
    if not user_repos or not corpus.repos:
        return Report(user, (), (), corpus, len(user_repos), budget)

    user_vectors = _profile_vectors(embedder, user_repos)
    corpus_vectors = _profile_vectors(embedder, list(corpus.repos))

    # For each corpus repo, how close is the user's nearest work?
    nearest: list[tuple[float, int]] = []
    for vector in corpus_vectors:
        best_score, best_index = -1.0, 0
        for index, own in enumerate(user_vectors):
            score = cosine(vector, own)
            if score > best_score:
                best_score, best_index = score, index
        nearest.append((best_score, best_index))

    # Grouped on a squashed key so `style-guide` and `styleguide` are one row
    # rather than two identical ones, but displayed using whichever spelling
    # the corpus used most - GitHub topics are user-chosen and both forms are
    # in circulation.
    by_key: dict[str, list[int]] = defaultdict(list)
    spellings: dict[str, Counter[str]] = defaultdict(Counter)
    for index, repo in enumerate(corpus.repos):
        for topic in repo.topics:
            cleaned = topic.strip().lower()
            if not cleaned or cleaned in GENERIC_TOPICS:
                continue
            key = re.sub(r"[^a-z0-9]", "", cleaned)
            if not key:
                continue
            by_key[key].append(index)
            spellings[key][cleaned] += 1

    by_topic = {
        spellings[key].most_common(1)[0][0]: indices for key, indices in by_key.items()
    }

    # The distribution this run's numbers are ranked against. See the note on
    # GAP_PERCENTILE for why this is relative rather than absolute.
    all_scores = sorted(score for score, _ in nearest)

    def percentile_of(value: float) -> float:
        below = sum(1 for score in all_scores if score < value)
        return below / len(all_scores)

    gaps: list[Gap] = []
    strengths: list[Strength] = []
    for topic, indices in by_topic.items():
        if len(indices) < MIN_PREVALENCE:
            continue
        # Coverage is the *best* the portfolio manages against this topic. Using
        # the mean would let one distant repository drag a covered topic into
        # the gap list, which reads as "you have never touched this" about
        # something the user has built.
        best_score, best_corpus_index = max((nearest[i][0], i) for i in indices)
        closest_own = user_repos[nearest[best_corpus_index][1]]
        rank = percentile_of(best_score)

        if rank >= STRENGTH_PERCENTILE:
            strengths.append(Strength(topic, best_score, closest_own.full_name))
            continue
        if rank > GAP_PERCENTILE:
            continue  # neither notably near nor notably far - say nothing

        evidence = sorted(
            (corpus.repos[i] for i in indices), key=lambda r: r.stars, reverse=True
        )
        gaps.append(
            Gap(
                topic=topic,
                coverage=best_score,
                prevalence=len(indices),
                evidence=tuple(evidence),
                nearest_repo=closest_own.full_name,
                percentile=rank,
            )
        )

    gaps.sort(key=lambda g: g.score, reverse=True)
    strengths.sort(key=lambda s: s.coverage, reverse=True)
    return Report(
        user=user,
        gaps=tuple(gaps[:limit]),
        strengths=tuple(strengths[:3]),
        corpus=corpus,
        repos_analysed=len(user_repos),
        budget=budget,
    )
