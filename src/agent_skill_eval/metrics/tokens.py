"""Token/cost accounting for trajectories.

Real numbers (``result`` event usage, ``total_cost_usd``) are always preferred;
estimates (chars/4, PRICING table) are used only when they are absent and are
flagged as estimated everywhere they surface.
"""

from __future__ import annotations

from agent_skill_eval import config
from agent_skill_eval.models.trajectory import Trajectory, Usage


def estimate_tokens(text_or_chars: str | int) -> int:
    chars = text_or_chars if isinstance(text_or_chars, int) else len(text_or_chars)
    return max(1, chars // config.CHARS_PER_TOKEN) if chars else 0


def estimate_cost(usage: Usage, model: str | None) -> float:
    input_rate, output_rate = config.PRICING.get(model or "", config.DEFAULT_PRICING)
    cost = (
        usage.input_tokens * input_rate
        + usage.cache_read_input_tokens * input_rate * config.CACHE_READ_MULTIPLIER
        + usage.cache_creation_input_tokens * input_rate * config.CACHE_CREATION_MULTIPLIER
        + usage.output_tokens * output_rate
    )
    return cost / 1_000_000


def trajectory_usage(trajectory: Trajectory) -> tuple[Usage, bool]:
    """Best-available run usage; second value is True when it was estimated."""
    if trajectory.outcome.usage is not None:
        return trajectory.outcome.usage, False
    total = Usage()
    saw_real = False
    for turn in trajectory.turns:
        if turn.usage is not None:
            saw_real = True
            total.output_tokens += turn.usage.output_tokens
            total.input_tokens = max(total.input_tokens, turn.usage.input_tokens)
            total.cache_read_input_tokens = max(
                total.cache_read_input_tokens, turn.usage.cache_read_input_tokens
            )
            total.cache_creation_input_tokens += turn.usage.cache_creation_input_tokens
    if saw_real:
        return total, False
    # Pure estimate: everything that crossed the wire, chars/4.
    chars = sum(t.char_len for t in trajectory.turns) + sum(trajectory.inter_turn_char_lens)
    total.output_tokens = sum(estimate_tokens(t.char_len) for t in trajectory.turns)
    total.input_tokens = max(0, estimate_tokens(chars) - total.output_tokens)
    return total, True


def trajectory_cost(trajectory: Trajectory) -> tuple[float, bool]:
    """Best-available run cost in USD; second value is True when estimated."""
    if trajectory.outcome.total_cost_usd is not None:
        return trajectory.outcome.total_cost_usd, False
    usage, _ = trajectory_usage(trajectory)
    return estimate_cost(usage, trajectory.model), True
