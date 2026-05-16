---
name: carts-simplify
description: Use when finalizing CARTS code, tests, skills, docs, or commits; after compiler pass or runtime edits; or when reducing patch complexity, checking utility placement, removing duplication, reusing existing APIs, or enforcing .carts artifact discipline.
---

# CARTS Simplify

Run this as the mandatory final simplification gate after implementation and
before commit, PR review, or final task completion.

## Workflow

1. Inspect the actual patch:
   ```bash
   git diff --stat
   git diff
   ```
2. State the intended behavior change in one sentence.
3. Confirm this is a production solution:
   - root cause is understood;
   - the changed dialect is the semantic owner;
   - the solution respects that dialect's function and limits;
   - no correctness depends on a downstream band-aid.
4. Remove accidental complexity:
   - dead code, unused helpers, stale comments, debug prints, and speculative
     abstractions;
   - unrelated refactors that are not required for correctness;
   - generated diagnostics, dumps, logs, and scratch files that are outside
     `.carts/` or are not intentional fixtures.
5. Check reuse before keeping new helpers:
   ```bash
   rg "<function-or-concept>" include/carts lib/carts docs
   rg "static .*<concept>|<attribute>|<predicate>" lib/carts include/carts
   ```
   Use `carts-check-utils` when a helper, predicate, string utility, IR query,
   or attribute helper was added or changed.
6. Put shared behavior in the narrowest correct home:
   - pass-local only when truly local to one transform and not useful to a
     verifier, analysis, conversion, or sibling pass;
   - dialect `Utils/` when it expresses a dialect invariant or reusable
     dialect-local builder/query/predicate;
   - `include/carts/utils/` and `lib/carts/utils/` when broadly meaningful;
   - analysis APIs when the logic belongs to DB/EDT/loop/cache/metadata state.
7. Re-check CARTS boundaries:
   - no hardcoded project attribute strings in pass logic;
   - no direct analysis graph mutation outside owning analysis code;
   - no pass-order assumptions hidden in tests or fixtures;
   - dependent dialect registration follows nearby pass patterns when a pass
     creates ops from another dialect;
   - lit/e2e coverage lands at the earliest stable owning stage;
   - canonicalization, folding, and cleanup passes are not required for
     correctness.
8. Run the smallest meaningful verification for the touched surface, then
   broaden only when blast radius requires it.

## Compiler-Safe Simplification

For MLIR changes, simplify by moving logic to the right abstraction, not by
making later cleanup passes responsible for correctness:

- prefer op `fold` or local canonicalization patterns only for convergent,
  local rewrites;
- put malformed IR checks in verifiers when they express dialect invariants;
- keep operation passes isolated to their operation scope and avoid global
  mutable state;
- preserve analyses only when the transform actually preserves them;
- call `signalPassFailure()` when a required invariant cannot be maintained.

## Artifact Discipline

Use repo-local scratch paths for development outputs when a command supports an
output location:

- `.carts/outputs/<topic>/...` for logs, diagnostics, stage dumps, benchmark
  triage, generated executables, and one-off command output.
- `.carts/sessions/<YYYYMMDD-HHMMSS>-<topic>/...` for multi-step debugging or
  investigation sessions that need several related artifacts.

Do not use `/tmp` in CARTS skills or final instructions unless an external tool
requires it. Do not commit `.carts/` artifacts unless promoting a reduced case
into an intentional fixture.

## Completion Rule

Do not claim the change is ready until this pass either made the patch smaller
or clearer, or explicitly found nothing worth simplifying.
