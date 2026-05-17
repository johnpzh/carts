---
name: carts-finishing
description: Use when advancing, continuing, or finishing CARTS work; choosing the next fix; asking where a fix belongs; applying regression guards; or referencing the carts-finishing plan.
user-invocable: true
allowed-tools: Bash, Read, Edit, Write, Grep, Glob, Agent, Skill, TaskList, TaskGet, TaskCreate, TaskUpdate, ScheduleWakeup, AskUserQuestion
argument-hint: [next | status | charter | triage <symptom> | regression-guard]
---

# CARTS Finishing

Project orchestrator for the CARTS finish-line plan. Tracks the 11-phase task graph, picks the next actionable item, delegates the actual fix work to focused triage skills, and enforces a regression-guard before advancing.

## What this skill does (and does not)

**Does:**
- Decide which phase / task to advance.
- Apply the operating principle: *symptom stage ≠ fix stage*.
- Walk the verification-barrier ladder to bound where a regression originated.
- Enforce the regression-guard checklist before marking a task done.
- Delegate execution to the right specialized triage skill.

**Does not:**
- Replace `carts-miscompile-triage`, `carts-distributed-triage`, `carts-stage-diff`, `carts-analysis-triage`, `carts-debug`, or `carts-runtime-triage`. This skill *invokes* those skills.
- Duplicate the project docs. It cites durable docs under `docs/compiler/`
  plus session archives under `.carts/sessions/...` and updates the skill if
  they diverge.

## The operating principle

**Errors usually surface several stages downstream of where they were introduced.** Fixing in the surface layer creates silent miscompilation. Recent CARTS history has at least three documented cases of this anti-pattern (see `references/triage-rubric.md` "Anti-patterns from prior fixes").

Three rules every iteration:
1. Walk the **verification-barrier ladder** to bound the regression window.
2. Identify the **originating layer** using the triage decision tree.
3. Run the **regression-guard checklist** before advancing.

## Iteration entry point

When invoked (default action `next`):

1. **Read state.** `TaskList` to see the current phase. If no carts-finishing tasks exist, stop and tell the user to seed them (the original plan synthesis creates tasks #1–11).
2. **Pick the next task.** Lowest-ID `pending` task with empty `blockedBy`. If everything is `in_progress`, resume that one. Mark `in_progress` via `TaskUpdate`.
3. **Identify the phase.** Tasks 1–11 correspond to phases 0–10 (see `references/phase-plan.md`). Each phase has a stop condition; do not advance until it is met.
4. **Execute the phase.** See "Phase dispatch" below.
5. **Run regression-guard.** Always. Even on small fixes. (`references/regression-guard.md`)
6. **Append a fix-attribution row.** If a code change landed, append a row to `docs/compiler/fix-attribution-log.md` (create the file on first use) with `symptom-stage | fix-stage | sample-or-benchmark | one-line-rationale`.
7. **Update task state.** `TaskUpdate` to `completed` if the stop condition is met; otherwise stay `in_progress` and add a brief metadata note.
8. **Schedule next iteration** via `ScheduleWakeup` if running under `/loop` dynamic mode. Use a 1200–1800s heartbeat (cache-window aware). Pass the same `/loop` input verbatim.

## Phase dispatch

| Phase | Type | What to do |
|---|---|---|
| 0 (task #1) | decision | Use `AskUserQuestion` for the 5 open charter questions in `references/dialect-charter.md` "Open questions". Save answers under `.carts/sessions/<topic>/charter-decisions.md`. |
| 1 (#2) | baseline | Run `dekk carts test`, `dekk carts test --suite e2e`, `dekk carts benchmarks run --size small`. Snapshot to `.results/baseline-YYYY-MM-DD.json`. Add multinode lit rule per `references/phase-plan.md` Phase 1. |
| 2 (#3) | retired diagnostic | The monolithic partitioning heuristic pass is gone. If metadata-copy recursion or DB-mode churn reappears, inspect the live `copyArtsMetadataAttrs` call sites, `DbAnalysis`, and `DbTransformsPass`; do not recreate the retired partitioning layer. |
| 3 (#4) | targeted-fix | Edit `lib/carts/dialect/sde/Conversion/PolygeistToSde/MemrefNormalization.cpp` first; `CreateDbs.cpp` is only the ARTS coarse raw bridge and boundary guard. The fix is in shape normalization, NOT in DB partitioning. See `references/triage-rubric.md` anti-pattern #1. |
| 4 (#5) | per-item iteration | One sample at a time. Use the per-item workflow below. |
| 5 (#6) | targeted-fix | Keep SDE distribution planning target-neutral. Add or adjust distributed eligibility in ARTS DB ownership/refinement so ARTS consumes SDE/CODIR contracts and abstract-machine topology without pushing topology decisions back into SDE or CODIR. |
| 6 (#7) | per-item iteration | One sample at a time, multinode. See `references/multinode-failures.md` before opening any file. |
| 7 (#8) | baseline | Re-run benchmark suite single-node. Document each regression vs the 2026-03-11 snapshot. |
| 8 (#9) | per-item iteration | One benchmark at a time, multinode. Same workflow as Phase 6. |
| 9 (#10) | structural | Pass-placement cleanup. Sub-steps in `references/phase-plan.md`. **Do not start until phase 8 is green.** |
| 10 (#11) | docs | Update docs, merge branch. |

## Per-item workflow (phases 4, 6, 8)

For each failing sample or benchmark:

1. **Compile + run.** Capture full error output and the failing pipeline stage.
2. **Walk the barrier ladder.** Run `dekk carts compile <file> --all-pipelines -o .carts/outputs/stages/<topic>/`. Find the first failing verify-barrier (see `references/triage-rubric.md` "Verification barrier map"). The bug lives between the previous green barrier and the failing one.
3. **Apply the decision tree.** Match symptom against `references/triage-rubric.md` "Triage decision tree". Identify the originating layer.
4. **Delegate.** Invoke the matching triage skill via `Skill`:
   - Wrong output / silent miscompilation → `carts-miscompile-triage`
   - Failure only with `--distributed-db` or multiple nodes → `carts-distributed-triage`
   - Behavior depends on pass order, stale graphs, metadata inconsistency → `carts-analysis-triage`
   - Crash, segfault, generic compile error → `carts-debug`
   - Need to compare two stages → `carts-stage-diff`
5. **Apply the fix in the originating layer**, not the surface layer.
6. **Regression-guard.** Run the 10-item checklist. Re-run all prior-green samples in BOTH single-node and multinode.
7. **Append fix-attribution log.** `docs/compiler/fix-attribution-log.md`.
8. **Move to next item.** Do not advance with regressions outstanding.

If the per-item workflow keeps failing on a single item across multiple iterations: stop, escalate to the user with a concrete summary of what's been tried and what blocked each attempt. Do not flail.

## Argument shortcuts

- `next` (default) — execute the next iteration as above.
- `status` — print `TaskList` summary + identify which phase we're in + last fix-attribution log entry.
- `charter` — read `references/dialect-charter.md` and answer "where does X belong?" using the rubric.
- `triage <symptom>` — read `references/triage-rubric.md` and identify the most likely originating layer for the described symptom.
- `regression-guard` — run the 10-item checklist now without doing anything else.

## Source-of-truth docs (project-level)

This skill cites but does not duplicate:

- `docs/compiler/pipeline.md` — live pipeline documentation; verify with `dekk carts pipeline --json`
- `docs/compiler/dialects/*` — dialect ownership docs for SDE, CODIR, ARTS, and ARTS-RT
- `.carts/sessions/...` — planning, experiments, investigation notes, and benchmark evidence
- `tools/compile/Compile.cpp` plus ARTS DB/ownership implementation anchors — current failure taxonomy lives with the pipeline and refinement code

**If any project doc disagrees with this skill, the project doc wins.** Update the skill to match. The references in this skill are distilled rubrics meant to be fast to consult during iteration; the project docs are authoritative.

## References

- `references/dialect-charter.md` — three-dialect charter, hard invariants, known violations, open questions
- `references/triage-rubric.md` — decision tree, verification barriers, stage→file map, anti-patterns
- `references/regression-guard.md` — the 10-item checklist
- `references/multinode-failures.md` — multinode-only failure modes (single-node tests will not catch these)
- `references/phase-plan.md` — phase descriptions, stop conditions, delegation map, sequencing rationale
