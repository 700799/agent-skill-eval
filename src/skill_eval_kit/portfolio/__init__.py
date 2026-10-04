"""Portfolio subsystem: audit, rank, and critique an entire skill library."""

from skill_eval_kit.portfolio.critic import critique_all, critique_skill
from skill_eval_kit.portfolio.lint import lint_all, lint_skill
from skill_eval_kit.portfolio.mine import mine_sessions
from skill_eval_kit.portfolio.scorecard import ab_score, build_scorecard, composite_score
from skill_eval_kit.portfolio.skills import discover_skills, parse_skill
from skill_eval_kit.portfolio.tfidf import cluster_overlaps, similarity_matrix
from skill_eval_kit.portfolio.triggers import (
    cross_activation_pairs,
    load_probes,
    run_triggers,
)

__all__ = [
    "ab_score",
    "build_scorecard",
    "cluster_overlaps",
    "composite_score",
    "critique_all",
    "critique_skill",
    "cross_activation_pairs",
    "discover_skills",
    "lint_all",
    "lint_skill",
    "load_probes",
    "mine_sessions",
    "parse_skill",
    "run_triggers",
    "similarity_matrix",
]
