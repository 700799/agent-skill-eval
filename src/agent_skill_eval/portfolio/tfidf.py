"""Hand-rolled TF-IDF similarity and clustering (no heavyweight deps).

Used to find skills that overlap enough to be merge candidates — the core
signal behind a MERGE recommendation in a bloated skill library.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from agent_skill_eval import config
from agent_skill_eval.models.portfolio import OverlapCluster, SkillMeta

_WORD = re.compile(r"[a-z0-9_]+")

STOPWORDS = frozenset(
    """a an the and or but if then else when while for to of in on at by with from as is are was
    were be been being do does did doing have has had having this that these those it its you your
    they them their we our us not no yes can will would should could may might must use used using
    skill claude code also into out up down over under more most other some such only own same
    than too very just about after before during each few both all any""".split()  # noqa: SIM905
)


def tokenize(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in STOPWORDS and len(t) > 2]


def _tf(tokens: list[str]) -> dict[str, float]:
    counts = Counter(tokens)
    return {term: 1.0 + math.log(count) for term, count in counts.items()}


def build_vectors(documents: dict[str, str]) -> dict[str, dict[str, float]]:
    """L2-normalized TF-IDF vectors, keyed like the input documents."""
    tokenized = {key: tokenize(text) for key, text in documents.items()}
    n_docs = len(tokenized) or 1
    doc_freq: Counter[str] = Counter()
    for tokens in tokenized.values():
        doc_freq.update(set(tokens))

    vectors: dict[str, dict[str, float]] = {}
    for key, tokens in tokenized.items():
        weights: dict[str, float] = {}
        for term, tf in _tf(tokens).items():
            idf = math.log((1 + n_docs) / (1 + doc_freq[term])) + 1.0
            weights[term] = tf * idf
        norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
        vectors[key] = {term: w / norm for term, w in weights.items()}
    return vectors


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(weight * b.get(term, 0.0) for term, weight in a.items())


def similarity_matrix(skills: list[SkillMeta]) -> dict[tuple[str, str], float]:
    documents = {s.name: f"{s.description}\n{s.body}" for s in skills}
    vectors = build_vectors(documents)
    names = sorted(vectors)
    matrix: dict[tuple[str, str], float] = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            matrix[(left, right)] = cosine(vectors[left], vectors[right])
    return matrix


def cluster_overlaps(
    skills: list[SkillMeta],
    threshold: float = config.DEFAULT_SIMILARITY_THRESHOLD,
) -> list[OverlapCluster]:
    """Union-find clusters of skills whose pairwise similarity meets ``threshold``."""
    matrix = similarity_matrix(skills)
    parent: dict[str, str] = {s.name: s.name for s in skills}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    best: dict[str, float] = {}
    for (left, right), score in matrix.items():
        if score < threshold:
            continue
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left
        merged = find(left)
        best[merged] = max(best.get(merged, 0.0), score)

    groups: dict[str, list[str]] = {}
    for name in parent:
        groups.setdefault(find(name), []).append(name)

    clusters: list[OverlapCluster] = []
    for root, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        clusters.append(
            OverlapCluster(
                skills=sorted(members),
                max_similarity=round(best.get(root, threshold), 4),
                rationale=(
                    "descriptions and bodies overlap heavily; one skill can likely "
                    "absorb the others"
                ),
            )
        )
    return clusters
