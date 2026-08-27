"""Portfolio subsystem: discovery, lint, clustering, triggers, mining, scorecard."""

from pathlib import Path

import pytest
from tests.conftest import FakeLLM

from skill_eval_kit.models.portfolio import (
    CriticReport,
    LintReport,
    MinedSkillUsage,
    MiningReport,
    SkillMeta,
    TriggerReport,
)
from skill_eval_kit.models.results import ABReport, ArmAggregate
from skill_eval_kit.portfolio import (
    build_scorecard,
    cluster_overlaps,
    composite_score,
    critique_all,
    discover_skills,
    lint_all,
    load_probes,
    mine_sessions,
    run_triggers,
    similarity_matrix,
)
from skill_eval_kit.portfolio.scorecard import ab_score, coverage_confidence, shrink
from skill_eval_kit.portfolio.triggers import cross_activation_pairs, dump_probe_template

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def example_skills() -> list[SkillMeta]:
    return discover_skills(EXAMPLES / "skills")


def arm(
    name: str, *, pass_rate: float, tokens: float, activation: float = 1.0
) -> ArmAggregate:
    return ArmAggregate(
        arm=name,  # type: ignore[arg-type]
        n=2,
        pass_rate=pass_rate,
        mean_total_tokens=tokens,
        median_total_tokens=tokens,
        mean_cost_usd=tokens / 1e6,
        mean_duration_seconds=1.0,
        mean_duplicate_calls=0.0,
        mean_re_reads=0.0,
        skill_activation_rate=activation,
    )


def ab_for(
    skill: str,
    *,
    control_pass: float = 0.0,
    treatment_pass: float = 1.0,
    control_tokens: float = 20000,
    treatment_tokens: float = 10000,
    activation: float = 1.0,
) -> ABReport:
    control = arm("control", pass_rate=control_pass, tokens=control_tokens, activation=0.0)
    treatment = arm(
        "treatment", pass_rate=treatment_pass, tokens=treatment_tokens, activation=activation
    )
    return ABReport(
        task_id=f"task-{skill}",
        skill_name=skill,
        control=control,
        treatment=treatment,
        deltas={"pass_rate_delta": treatment_pass - control_pass},
    )


class TestDiscoveryAndLint:
    def test_discovers_example_library(self, example_skills: list[SkillMeta]) -> None:
        assert [s.name for s in example_skills] == [
            "fastapi-schema",
            "helpful-helper",
            "py-validation",
        ]
        good = next(s for s in example_skills if s.name == "fastapi-schema")
        assert "Pydantic v2" in good.description
        assert good.token_estimate > 0

    def test_flat_md_layout(self, tmp_path: Path) -> None:
        (tmp_path / "solo.md").write_text("---\nname: solo\n---\nbody\n", encoding="utf-8")
        assert [s.name for s in discover_skills(tmp_path)] == ["solo"]

    def test_lint_separates_good_from_vague(self, example_skills: list[SkillMeta]) -> None:
        by_skill = {r.skill: r for r in lint_all(example_skills)}
        assert by_skill["fastapi-schema"].score == 1.0
        assert by_skill["fastapi-schema"].findings == []

        vague = by_skill["helpful-helper"]
        checks = {f.check for f in vague.findings}
        assert {"vague-description", "dead-reference", "no-examples"} <= checks
        assert vague.score < 0.5

    def test_token_footprint_findings(self, tmp_path: Path) -> None:
        bundle = tmp_path / "fat"
        bundle.mkdir()
        body = "word " * 20_000  # ~25k chars -> ~6k tokens
        (bundle / "SKILL.md").write_text(
            f"---\nname: fat\ndescription: Use this skill when doing the fat thing repeatedly.\n"
            f"---\n```py\nx=1\n```\n{body}",
            encoding="utf-8",
        )
        report = lint_all(discover_skills(tmp_path))[0]
        footprint = [f for f in report.findings if f.check == "token-footprint"]
        assert footprint and footprint[0].severity == "error"


class TestSimilarity:
    def test_overlapping_pair_clusters(self, example_skills: list[SkillMeta]) -> None:
        matrix = similarity_matrix(example_skills)
        assert matrix[("fastapi-schema", "py-validation")] > 0.45
        assert matrix[("fastapi-schema", "helpful-helper")] < 0.1

        clusters = cluster_overlaps(example_skills)
        assert len(clusters) == 1
        assert clusters[0].skills == ["fastapi-schema", "py-validation"]

    def test_threshold_is_tunable(self, example_skills: list[SkillMeta]) -> None:
        assert cluster_overlaps(example_skills, threshold=0.99) == []


class TestTriggers:
    def test_classifier_scoring_and_confusion(self, example_skills: list[SkillMeta]) -> None:
        probes = load_probes(EXAMPLES / "probes.yaml")
        # Scripted: fastapi-schema is precise; py-validation also fires on
        # fastapi-schema's probes (cross-activation); helpful-helper never fires.
        scripted = []
        for text, intended in [
            ("Add a response model with validation to my FastAPI endpoint.", "fastapi-schema"),
            ("This route returns a raw dict — give it a proper schema.", "fastapi-schema"),
            ("My Pydantic model needs a custom validator for the email field.", "py-validation"),
            ("Help me with the thing in the usual way.", "helpful-helper"),
        ]:
            del text
            if intended == "fastapi-schema":
                scripted.append({"activations": ["fastapi-schema", "py-validation"]})
            elif intended == "py-validation":
                scripted.append({"activations": ["py-validation"]})
            else:
                scripted.append({"activations": []})
        scripted.extend([{"activations": []}] * len(probes.shared_negatives))

        llm = FakeLLM(scripted)
        reports, confusion = run_triggers(
            probes, example_skills, mode="classifier", llm=llm
        )
        by_skill = {r.skill: r for r in reports}

        fastapi = by_skill["fastapi-schema"]
        assert fastapi.tp == 2 and fastapi.fn == 0
        assert fastapi.precision == 1.0 and fastapi.recall == 1.0

        pyval = by_skill["py-validation"]
        assert pyval.tp == 1
        assert pyval.fp == 2  # fired on fastapi-schema's two probes
        assert pyval.precision < 1.0

        helper = by_skill["helpful-helper"]
        assert helper.recall == 0.0 and helper.f1 == 0.0

        assert confusion["fastapi-schema"]["py-validation"] == 2
        assert confusion["(negative)"]["(none)"] == len(probes.shared_negatives)

        pairs = cross_activation_pairs(confusion)
        assert ("fastapi-schema", "py-validation", 0.5) in pairs

    def test_unknown_skill_in_probes_rejected(self, example_skills: list[SkillMeta]) -> None:
        probes = load_probes(EXAMPLES / "probes.yaml")
        with pytest.raises(ValueError, match="unknown skills"):
            run_triggers(probes, example_skills[:1], mode="classifier", llm=FakeLLM([]))

    def test_classifier_mode_needs_llm(self, example_skills: list[SkillMeta]) -> None:
        probes = load_probes(EXAMPLES / "probes.yaml")
        with pytest.raises(RuntimeError, match="LLM caller"):
            run_triggers(probes, example_skills, mode="classifier", llm=None)

    def test_probe_template_covers_every_skill(self, example_skills: list[SkillMeta]) -> None:
        template = dump_probe_template(example_skills)
        for meta in example_skills:
            assert meta.name in template


class TestCriticAndMining:
    def test_critic_reports(self, example_skills: list[SkillMeta]) -> None:
        llm = FakeLLM(
            [
                {"score": 0.9, "issues": [], "rewrite_suggestions": []},
                {
                    "score": 0.2,
                    "issues": ["description says nothing"],
                    "rewrite_suggestions": ["name concrete triggers"],
                    "improved_description": "Use when ...",
                },
                {"score": 0.6, "issues": ["overlaps fastapi-schema"], "rewrite_suggestions": []},
            ]
        )
        reports = critique_all(example_skills, llm)
        assert [r.skill for r in reports] == [s.name for s in example_skills]
        assert reports[1].score == 0.2
        assert reports[1].improved_description == "Use when ..."

    def test_critic_failure_is_a_datum(self, example_skills: list[SkillMeta]) -> None:
        reports = critique_all(example_skills[:1], FakeLLM([]))
        assert reports[0].score == 0.0
        assert "critic call failed" in reports[0].issues[0]

    def test_mining_counts_and_zero_usage(self, mined_fixture: Path) -> None:
        report = mine_sessions(
            [mined_fixture.parent],
            known_skills=["fastapi-schema", "py-validation", "helpful-helper"],
        )
        assert report.sessions_scanned == 1
        by_skill = {u.skill: u for u in report.skills}
        assert by_skill["fastapi-schema"].activations == 1
        assert by_skill["fastapi-schema"].est_cost_usd > 0
        # skills that never fired are reported at zero — that is the RETIRE signal
        assert by_skill["py-validation"].activations == 0


class TestScorecard:
    def test_composite_renormalizes_missing_tiers(self) -> None:
        # Only lint present -> composite is just the lint score.
        assert composite_score({"lint": 0.8, "trigger": None, "ab": None, "critic": None}) == 0.8
        # lint + ab -> weights 0.15 and 0.40 renormalized to 0.273 / 0.727
        mixed = composite_score({"lint": 1.0, "trigger": None, "ab": 0.0, "critic": None})
        assert abs(mixed - 0.15 / 0.55) < 1e-9
        assert composite_score({"lint": None, "trigger": None, "ab": None, "critic": None}) == 0.0

    def test_ab_score_rewards_lift_savings_adherence(self) -> None:
        outcome = ab_score([ab_for("s")])
        assert outcome is not None
        score, parts = outcome
        assert parts["pass_lift"] == 1.0
        assert parts["token_savings"] == 1.0
        assert parts["adherence"] == 1.0
        assert score == 1.0
        assert ab_score([]) is None

    def test_keep_fix_merge_retire_all_reachable(
        self, example_skills: list[SkillMeta]
    ) -> None:
        lint = lint_all(example_skills)
        clusters = cluster_overlaps(example_skills)
        triggers = [
            TriggerReport(skill="fastapi-schema", precision=1.0, recall=1.0, f1=1.0, tp=2),
            TriggerReport(skill="py-validation", precision=0.5, recall=1.0, f1=0.67, tp=1, fp=2),
            TriggerReport(skill="helpful-helper", precision=0.0, recall=0.0, f1=0.0, fn=1),
        ]
        critics = [
            CriticReport(skill="fastapi-schema", score=0.95),
            CriticReport(skill="py-validation", score=0.6),
            CriticReport(skill="helpful-helper", score=0.1),
        ]
        abs_ = [
            ab_for("fastapi-schema"),
            ab_for("py-validation", treatment_pass=1.0, treatment_tokens=19000),
            # no lift, no savings, never activates: the RETIRE signature
            ab_for(
                "helpful-helper",
                control_pass=0.5,
                treatment_pass=0.5,
                treatment_tokens=26000,
                activation=0.0,
            ),
        ]
        mining = MiningReport(
            skills=[
                MinedSkillUsage(skill="fastapi-schema", activations=12),
                MinedSkillUsage(skill="py-validation", activations=3),
                MinedSkillUsage(skill="helpful-helper", activations=0),
            ]
        )
        report = build_scorecard(
            example_skills,
            lint_reports=lint,
            trigger_reports=triggers,
            ab_reports=abs_,
            critic_reports=critics,
            mining=mining,
            clusters=clusters,
            cross_activation=[("fastapi-schema", "py-validation", 0.5)],
        )
        by_skill = {e.skill: e for e in report.entries}

        assert by_skill["fastapi-schema"].recommendation == "KEEP"
        assert by_skill["py-validation"].recommendation == "MERGE"
        assert by_skill["py-validation"].merge_with == ["fastapi-schema"]
        assert by_skill["helpful-helper"].recommendation == "RETIRE"

        # sorted by composite, best first; every entry explains itself
        assert report.entries[0].skill == "fastapi-schema"
        assert all(e.reasons for e in report.entries)
        assert report.clusters[0].merge_into == "fastapi-schema"

    def test_evidence_outranks_absence_of_evidence(self) -> None:
        """A measured-good skill must not lose to an unmeasured one with clean lint."""
        measured = SkillMeta(name="measured", path="m.md", description="d", body="b")
        unmeasured = SkillMeta(name="unmeasured", path="u.md", description="d", body="b")
        report = build_scorecard(
            [measured, unmeasured],
            lint_reports=[
                LintReport(skill="measured", score=1.0),
                LintReport(skill="unmeasured", score=1.0),
            ],
            ab_reports=[ab_for("measured")],
        )
        by_skill = {e.skill: e for e in report.entries}
        assert report.entries[0].skill == "measured"
        assert by_skill["measured"].confidence > by_skill["unmeasured"].confidence
        assert by_skill["measured"].ranked_score > by_skill["unmeasured"].ranked_score
        # the raw composites are equal — only evidence breaks the tie
        assert by_skill["measured"].composite == by_skill["unmeasured"].composite

    def test_confidence_and_shrinkage(self) -> None:
        lint_only = {"lint": 1.0, "trigger": None, "ab": None, "critic": None}
        assert coverage_confidence(lint_only) == pytest.approx(0.15)
        full = {"lint": 1.0, "trigger": 1.0, "ab": 1.0, "critic": 1.0}
        assert coverage_confidence(full) == pytest.approx(1.0)
        # full evidence is taken at face value; none is pinned to the prior
        assert shrink(0.9, 1.0) == pytest.approx(0.9)
        assert shrink(0.9, 0.0) == pytest.approx(0.5)
        assert shrink(1.0, 0.15) == pytest.approx(0.575)

    def test_lint_only_coverage_never_retires(self, example_skills: list[SkillMeta]) -> None:
        report = build_scorecard(example_skills, lint_reports=lint_all(example_skills))
        worst = {e.skill: e for e in report.entries}["helpful-helper"]
        assert worst.recommendation == "FIX"
        assert worst.coverage == {"lint": True, "trigger": False, "ab": False, "critic": False}

    def test_low_adherence_flags_fix(self) -> None:
        skill = SkillMeta(name="bypassed", path="bypassed.md", description="d", body="b")
        report = build_scorecard(
            [skill],
            lint_reports=[LintReport(skill="bypassed", score=1.0)],
            # tasks pass, tokens saved, but the skill is almost never used
            ab_reports=[ab_for("bypassed", activation=0.1)],
        )
        entry = report.entries[0]
        assert entry.recommendation == "FIX"
        assert any("adherence" in reason for reason in entry.reasons)
