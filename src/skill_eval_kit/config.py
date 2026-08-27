"""Central configuration: pricing, default models, scorecard weights, thresholds.

Prices are USD per million tokens (input, output), Anthropic first-party API
rates. Used only to *estimate* cost when the runner does not report
``total_cost_usd``; estimated costs are always flagged as such in results.
Cache reads are billed at 0.1x the input rate, cache creation at 1.25x.
"""

from __future__ import annotations

# model id -> (input $/MTok, output $/MTok)
PRICING: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

CACHE_READ_MULTIPLIER = 0.1
CACHE_CREATION_MULTIPLIER = 1.25

#: Fallback pricing for models absent from PRICING (opus-tier, conservative).
DEFAULT_PRICING: tuple[float, float] = (5.0, 25.0)

#: Rough token estimate used when per-message usage is absent from the stream.
CHARS_PER_TOKEN = 4

DEFAULT_JUDGE_MODEL = "claude-opus-5"
DEFAULT_CLASSIFIER_MODEL = "claude-haiku-4-5"
DEFAULT_CRITIC_MODEL = "claude-opus-5"

#: Portfolio scorecard component weights (renormalized over available tiers).
DEFAULT_WEIGHTS: dict[str, float] = {
    "lint": 0.15,
    "trigger": 0.25,
    "ab": 0.40,
    "critic": 0.20,
}

#: TF-IDF cosine similarity at or above which two skills join an overlap cluster.
#: Cosine on prose rarely exceeds ~0.6 even for genuine duplicates, while
#: unrelated skills sit near 0 — so 0.45 separates them with room to spare.
#: Tune per library size with `ase audit --similarity-threshold`.
DEFAULT_SIMILARITY_THRESHOLD = 0.45

#: Cross-activation fraction above which the confusion matrix signals a merge.
CROSS_ACTIVATION_MERGE_THRESHOLD = 0.30

#: Skill body token estimates above these thresholds draw lint findings
#: (the skill's own footprint is paid on every activation).
SKILL_TOKENS_WARNING = 2000
SKILL_TOKENS_ERROR = 4000

#: Files larger than this (in lines) count toward the re-read waste metric.
LARGE_FILE_LINES = 100
