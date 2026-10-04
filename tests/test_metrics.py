from pathlib import Path

from tests.conftest import TrajectoryBuilder

from skill_eval_kit.metrics import loops, tokens
from skill_eval_kit.models.trajectory import Usage
from skill_eval_kit.runner.ndjson import parse_file

BIG = {"file_path": "/ws/big.py"}


class TestDuplicates:
    def test_loopy_fixture_counts(self, loopy_fixture: Path) -> None:
        traj = parse_file(loopy_fixture, source="replay")
        report = loops.detect_duplicates(traj)
        # 4 byte-identical Reads of big_module.py -> 3 duplicates
        assert report.duplicate_tool_calls == 3
        assert report.worst_offenders[0]["tool"] == "Read"
        assert report.worst_offenders[0]["count"] == 4

    def test_exempt_tools_skipped(self) -> None:
        traj = (
            TrajectoryBuilder()
            .turn(("TodoWrite", {"todos": ["a"]}))
            .turn(("TodoWrite", {"todos": ["a"]}))
            .turn(("TodoWrite", {"todos": ["a"]}))
            .build()
        )
        assert loops.detect_duplicates(traj).duplicate_tool_calls == 0
        assert loops.detect_duplicates(traj, exempt_tools=[]).duplicate_tool_calls == 2

    def test_different_inputs_not_duplicates(self) -> None:
        traj = (
            TrajectoryBuilder()
            .turn(("Read", {"file_path": "/a.py"}))
            .turn(("Read", {"file_path": "/b.py"}))
            .build()
        )
        assert loops.detect_duplicates(traj).duplicate_tool_calls == 0


class TestReReads:
    def test_loopy_fixture_edit_resets_window(self, loopy_fixture: Path) -> None:
        traj = parse_file(loopy_fixture, source="replay")
        report = loops.detect_re_reads(traj)
        # reads 1-3 -> 2 re-reads; Edit resets; read 4 is first of a new window
        assert report.file_re_reads == 2
        assert report.per_path == {"/workspace/big_module.py": 2}

    def test_small_files_ignored(self) -> None:
        traj = (
            TrajectoryBuilder()
            .turn(("Read", {"file_path": "/s.py"}), result_lines=10)
            .turn(("Read", {"file_path": "/s.py"}), result_lines=10)
            .build()
        )
        assert loops.detect_re_reads(traj).file_re_reads == 0

    def test_write_resets_window(self) -> None:
        traj = (
            TrajectoryBuilder()
            .turn(("Read", BIG), result_lines=150)
            .turn(("Write", {"file_path": "/ws/big.py", "content": "x"}))
            .turn(("Read", BIG), result_lines=150)
            .build()
        )
        assert loops.detect_re_reads(traj).file_re_reads == 0

    def test_missing_result_assumed_large(self) -> None:
        traj = TrajectoryBuilder().turn(("Read", BIG)).turn(("Read", BIG)).build()
        assert loops.detect_re_reads(traj).file_re_reads == 1


class TestContextGrowth:
    def test_exact_from_usage(self) -> None:
        traj = (
            TrajectoryBuilder()
            .turn(text="a", usage=Usage(input_tokens=1000, cache_read_input_tokens=9000))
            .turn(text="b", usage=Usage(input_tokens=1000, cache_read_input_tokens=29000))
            .build()
        )
        report = loops.detect_context_growth(traj, threshold_tokens=15000)
        assert not report.estimated
        assert report.max_growth == 20000
        assert report.violations == [(1, 20000)]

    def test_first_turn_exempt(self) -> None:
        traj = (
            TrajectoryBuilder()
            .turn(text="a", usage=Usage(input_tokens=50000))
            .turn(text="b", usage=Usage(input_tokens=50500))
            .build()
        )
        assert loops.detect_context_growth(traj).violations == []

    def test_estimated_fallback(self, loopy_fixture: Path) -> None:
        traj = parse_file(loopy_fixture, source="replay")
        report = loops.detect_context_growth(traj, threshold_tokens=15000)
        assert report.estimated
        # the 70k-char blob before the final turn: ~17.5k estimated tokens
        assert report.max_growth > 15000
        assert report.violations and report.violations[-1][0] == len(traj.turns) - 1


class TestTokensAndCost:
    def test_real_usage_and_cost_preferred(self, success_fixture: Path) -> None:
        traj = parse_file(success_fixture, source="replay")
        usage, estimated = tokens.trajectory_usage(traj)
        assert not estimated
        assert usage.cache_read_input_tokens == 25800
        cost, cost_estimated = tokens.trajectory_cost(traj)
        assert (cost, cost_estimated) == (0.031, False)

    def test_estimated_cost_when_result_lacks_cost(self, error_fixture: Path) -> None:
        traj = parse_file(error_fixture, source="replay")
        cost, estimated = tokens.trajectory_cost(traj)
        assert estimated
        # sonnet-5: 900*2 + 30000*0.2 + 400*10 per MTok
        assert abs(cost - (900 * 2 + 30000 * 0.2 + 400 * 10) / 1e6) < 1e-9

    def test_estimate_cost_all_models(self) -> None:
        usage = Usage(input_tokens=1_000_000)
        assert tokens.estimate_cost(usage, "claude-opus-5") == 5.0
        assert tokens.estimate_cost(usage, "claude-sonnet-5") == 2.0
        assert tokens.estimate_cost(usage, "claude-haiku-4-5") == 1.0
        assert tokens.estimate_cost(usage, "claude-fable-5") == 10.0
        # unknown model falls back to opus-tier default
        assert tokens.estimate_cost(usage, "mystery-model") == 5.0

    def test_cache_multipliers(self) -> None:
        assert tokens.estimate_cost(
            Usage(cache_read_input_tokens=1_000_000), "claude-opus-5"
        ) == 0.5
        assert tokens.estimate_cost(
            Usage(cache_creation_input_tokens=1_000_000), "claude-opus-5"
        ) == 6.25

    def test_estimate_tokens(self) -> None:
        assert tokens.estimate_tokens("") == 0
        assert tokens.estimate_tokens("abcd" * 10) == 10
        assert tokens.estimate_tokens(2) == 1
