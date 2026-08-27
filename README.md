# skill-eval-kit

An evaluation framework for **Claude Code skills** and Python-dev (`pydev`) plugins.

Most skill "testing" answers only *did the task pass?* — which a capable model
often does anyway, skill or no skill. This framework measures whether a skill
actually **earns its keep**: does it change the outcome, does it save or waste
tokens, and did the agent genuinely follow it or just fall back on prior
knowledge and pass by luck?

It works at two scales:

- **One skill, deeply** — A/B a task with the skill loaded vs. not, and grade
  the result on code quality, trajectory efficiency, and architectural adherence.
- **A whole library** — audit, score, and rank every skill you have, with a
  KEEP / FIX / MERGE / RETIRE call for each. Built for teams whose skill
  collection has outgrown anyone's ability to review it by hand.

---

## Install

```bash
pip install skill-eval-kit        # installs the `skill-eval` command
```

From a checkout, for development:

```bash
uv venv && uv pip install -e ".[dev]"
```

Requires Python 3.11+. Live runs need the [Claude Code CLI](https://code.claude.com/docs)
on `PATH`; the LLM judge and critic need API credentials. **Everything else runs
offline** — pass `--no-llm` and `--runner replay` to work with no network at all.

## Quickstart

```bash
skill-eval validate examples/tasks/*.yaml examples/probes.yaml       # schema-check
skill-eval --no-llm audit examples/skills --json runs/audit.json     # Tier 0: lint + overlap
skill-eval mine ~/.claude/projects --skills-dir examples/skills --json runs/mine.json
skill-eval --no-llm ab examples/tasks/pydev-fastapi-pydantic-01.yaml \
    --runner replay --trials 2 \
    --control-fixture tests/fixtures/ndjson/loopy_no_skill.ndjson \
    --treatment-fixture tests/fixtures/ndjson/success_with_skill.ndjson
skill-eval rank examples/skills --audit runs/audit.json --runs runs \
    --mine runs/mine.json --md runs/leaderboard.md
```

Drop `--runner replay` and its fixtures to run against the real CLI.

---

## The four measurements

### 1. Task success & code quality

`pydev_static` runs **ruff** and **mypy** over the final workspace; `pydev_tests`
runs **pytest**; `ast_assertions` parses the resulting code with Python's own
**AST** module. Missing tools degrade to a recorded failure rather than a crash.

### 2. Token waste & loop detection

Three anti-patterns, computed from the trajectory:

| Detector | What counts | Why it is not naive |
|---|---|---|
| Duplicate tool calls | Same tool + byte-identical input | Canonical JSON hashing; exempt list for legitimately repeated tools (`TodoWrite`) |
| Large-file re-reads | Re-reading a >100-line file | An `Edit`/`Write` to that file **resets** the window — re-reading your own change is legitimate |
| Context growth | Tokens added per turn (default limit 15k) | Exact when the stream carries per-message usage; falls back to a chars/4 estimate and **flags that it did** |

### 3. Skill adherence vs. generic bypass

The framework's sharpest signal. Three independent checks:

1. **`skill_activation`** — did the `Skill` tool actually fire for the target skill?
2. **`ast_assertions`** — are the skill's prescribed constructs present and its
   forbidden ones absent (e.g. Pydantic v2 `@field_validator` required,
   v1 `@validator` banned)?
3. **`llm_judge`** — a rubric graded by Claude for what static analysis can't see.

A skill whose tasks pass while activation stays low is flagged **FIX**: the model
is solving it from prior knowledge and the skill is along for the ride.

### 4. A/B baseline differential

Every task runs in two arms:

| Arm | How | Meaning |
|---|---|---|
| **Control** | `claude -p --disable-slash-commands` | What the model does unaided |
| **Treatment** | Skill injected at `.claude/skills/<name>/SKILL.md` | What the skill adds |

Reports give `treatment − control` deltas for pass rate, tokens, cost, duration,
and each waste metric. Negative token/cost deltas mean the skill *saves*.

> The control arm uses `--disable-slash-commands`, **not** `--bare`. `--bare`
> only skips hooks, LSP, and plugin credentials — its help notes that "Skills
> still resolve via `/skill-name`", so a `--bare` baseline would silently load
> the very skill under test and understate its measured value.

---

## Ranking a whole library

Cost scales with how deeply you want to probe, so a large library can be triaged
before spending real money:

| Tier | Command | Cost | Produces |
|---|---|---|---|
| **0 · Static** | `skill-eval audit <dir>` | free | Lint findings + TF-IDF overlap clusters |
| **1 · Triggers** | `skill-eval audit <dir> --probes probes.yaml` | cheap | Precision/recall/F1 + cross-activation confusion matrix |
| **2 · A/B** | `skill-eval ab <task.yaml>` | real runs | Measured value lift |
| **Critic** | `skill-eval audit <dir> --critic` | one call/skill | Actionable rewrite suggestions |
| **Mining** | `skill-eval mine <transcripts>` | free | Real-world activation frequency and cost |

**Tier 0** catches the problems you can find for free: descriptions too vague to
route on, skills whose own body costs thousands of tokens on every activation,
dead file references, and near-duplicate skills that should be merged.

**Tier 1** asks whether a skill fires when it should — and stays quiet when it
shouldn't. Each skill's positive probes double as negatives for every *other*
skill, which fills a confusion matrix and exposes skills that keep firing on
each other's work (a merge signal independent of text similarity).

### Scoring

Composite score is a weighted mean over **whichever tiers have run**, with
weights renormalized across what's available — so a Tier 0-only library still
ranks, and the score sharpens as evidence accumulates.

```
lint 0.15 · trigger 0.25 · ab 0.40 · critic 0.20      (override with --weights)

ab = 0.50 × pass-rate lift + 0.25 × token savings + 0.25 × adherence
```

Recommendations, in precedence order:

- **RETIRE** — measured with no pass-rate lift, no token savings, and weak
  triggering; or a poor composite backed by real evidence; or never activated
  in mined sessions while scoring badly.
- **MERGE** — overlaps a higher-scoring sibling (TF-IDF cluster or
  cross-activation). `merge_with` names the survivor.
- **FIX** — blocking lint errors, trigger F1 below 0.70, a mid-range composite,
  or the bypass signature above.
- **KEEP** — clears every tier that has run.

**Confidence and ranking.** Renormalizing over available tiers alone would let a
skill with nothing but clean lint tie — or beat — one measured end to end and
found good. So each entry also carries a **confidence** (the share of scoring
weight actually backed by evidence), and *ranking* uses the composite shrunk
toward a neutral 0.5 in proportion to what's missing:

```
ranked = composite × confidence + 0.5 × (1 − confidence)
```

An unmeasured skill therefore cannot outrank a measured one, and the survivor of
a merge cluster is the best-*evidenced* member rather than the luckiest. The raw
composite is still what threshold decisions read, so the two roles stay separate.

**Guardrail:** a skill is never retired on static lint alone. With Tier 0 only,
the worst outcome is FIX, reason `insufficient_evidence`, and the leaderboard
prints `—` for every tier that didn't run. Thin evidence never impersonates a
measurement.

---

## Task schema

```yaml
id: pydev-fastapi-pydantic-01
description: "Verify the fastapi-schema skill refactors raw dicts into Pydantic v2 models."
target_skill: "../skills/fastapi-schema/SKILL.md"

workspace:
  initial_files:
    app.py: |
      from fastapi import FastAPI
      app = FastAPI()

prompt: "Refactor app.py to return a Pydantic model."

run_limits:
  max_turns: 6            # -> claude --max-turns
  max_budget_usd: 0.15    # -> claude --max-budget-usd
  # timeout_seconds, model, allowed_tools, disallowed_tools also supported

evaluations:
  - type: skill_activation
    expected: "fastapi-schema"
  - type: pydev_static
    linters: [ruff, mypy]
    strict: true
  - type: pydev_tests
  - type: ast_assertions
    assertions:
      - {file: app.py, kind: decorator_used, value: field_validator}
      - {file: app.py, kind: forbidden_decorator, value: validator}
  - type: trajectory_efficiency
    max_duplicate_tool_calls: 1
    max_file_re_reads: 2
  - type: llm_judge
    rubric: "Did the code use Pydantic V2 `@field_validator` syntax?"
```

The schema is strict: unknown fields and unknown evaluation types fail at load
time, not mid-run. `ast_assertions` kinds: `decorator_used`, `call_used`,
`import_used`, `class_inherits`, `function_defined`, and the `forbidden_*`
variants.

`probes.yaml` (for Tier 1) is documented in `examples/probes.yaml`.

---

## Architecture

Every measurement reads one normalized `Trajectory`, and three sources produce it:

```
claude -p --output-format stream-json ─┐
recorded NDJSON fixture ───────────────┼──> runner/ndjson.py ──> Trajectory ──> metrics
real session transcript (skill-eval mine) ────┘                                   └──> evaluators
```

Because replay is a first-class source, the entire pipeline — including the
ruff/mypy/pytest evaluators — is exercised in CI with no API key and no network.
`ReplayRunner` re-applies the `Write`/`Edit` calls recorded in a fixture, so an
offline run rebuilds the workspace the agent actually produced instead of
grading the starting files.

Cost comes from the CLI's reported `total_cost_usd` when present; otherwise it
is estimated from a pricing table and **marked as estimated** in every report.

## Development

```bash
uv run ruff check src tests
uv run mypy src
uv run pytest              # offline; live-marked tests are deselected by default
uv run pytest -m live      # needs an authenticated claude CLI, spends real tokens
python tests/fixtures/make_fixtures.py   # regenerate NDJSON fixtures
```

## Relationship to `claude plugin eval`

Claude Code ships an early-access, org-gated `claude plugin eval` that also runs
control/treatment arms with regex, tool-use, and LLM graders. This project is
standalone and complementary: it adds trajectory-level waste detection, AST
adherence checks, and portfolio-scale ranking, and it runs anywhere the CLI
runs. It is pinned against the `2.1.x` stream-json shape and parses defensively,
ignoring unknown event types and fields so format drift degrades rather than breaks.

## License

MIT
