from pathlib import Path

from agent_skill_eval.runner.ndjson import parse_events, parse_file


def test_success_fixture_parses(success_fixture: Path) -> None:
    traj = parse_file(success_fixture, source="replay")
    assert traj.model == "claude-sonnet-5"
    assert len(traj.turns) == 4
    assert traj.skill_activations() == ["fastapi-schema"]
    # every turn has per-message usage in this fixture
    assert all(t.usage is not None for t in traj.turns)
    # tool results matched by id
    calls = traj.iter_tool_calls()
    assert [c.name for c in calls] == ["Skill", "Read", "Write"]
    read_result = traj.result_for(calls[1])
    assert read_result is not None and "FastAPI" in read_result.content_text
    assert traj.outcome.is_success
    assert traj.outcome.total_cost_usd == 0.031
    assert traj.outcome.num_turns == 4
    assert traj.final_text().startswith("Done.")


def test_loopy_fixture_has_no_usage_and_no_skill(loopy_fixture: Path) -> None:
    traj = parse_file(loopy_fixture, source="replay")
    assert traj.skill_activations() == []
    assert all(t.usage is None for t in traj.turns)
    assert all(t.char_len > 0 for t in traj.turns)
    reads = [c for c in traj.iter_tool_calls() if c.name == "Read"]
    assert len(reads) == 5
    # inter-turn char accounting: the huge blob lands before the final turn
    assert len(traj.inter_turn_char_lens) == len(traj.turns)
    assert traj.inter_turn_char_lens[-1] > 70_000


def test_error_fixture_outcome(error_fixture: Path) -> None:
    traj = parse_file(error_fixture, source="replay")
    assert traj.outcome.subtype == "error_max_turns"
    assert not traj.outcome.is_success
    assert traj.outcome.total_cost_usd is None
    assert traj.outcome.usage is not None
    assert traj.outcome.usage.cache_read_input_tokens == 30000


def test_mined_session_parses_with_same_parser(mined_fixture: Path) -> None:
    traj = parse_file(mined_fixture, source="mined")
    assert traj.source == "mined"
    assert traj.skill_activations() == ["fastapi-schema"]
    assert traj.outcome.subtype == "missing_result"  # transcripts have no result event


def test_malformed_and_unknown_lines_skipped() -> None:
    lines = [
        "not json at all",
        '{"type": "unknown_event", "x": 1}',
        '{"type": "assistant", "message": {"id": "m1", "content": [{"type": "text", "text": "hi"}]}}',
        "",
        '{"type": "result", "subtype": "success", "num_turns": 1}',
    ]
    traj = parse_events(lines, source="replay")
    assert len(traj.turns) == 1
    assert traj.outcome.is_success


def test_split_assistant_events_merge_on_message_id() -> None:
    lines = [
        '{"type": "assistant", "message": {"id": "m1", "content": [{"type": "text", "text": "a"}]}}',
        '{"type": "assistant", "message": {"id": "m1", "content": '
        '[{"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "x.py"}}]}}',
        '{"type": "assistant", "message": {"id": "m2", "content": [{"type": "text", "text": "b"}]}}',
    ]
    traj = parse_events(lines, source="replay")
    assert len(traj.turns) == 2
    assert traj.turns[0].text == "a"
    assert [c.name for c in traj.turns[0].tool_calls] == ["Read"]


def test_missing_result_event() -> None:
    traj = parse_events(
        ['{"type": "assistant", "message": {"id": "m1", "content": []}}'], source="live"
    )
    assert traj.outcome.subtype == "missing_result"
    assert traj.outcome.num_turns == 1
