---
name: carts-commit
description: Use when staging CARTS changes, choosing a commit message scope, committing, pushing, or handing a patch to review.
---

# CARTS Commit

Use after implementation, verification, [[carts-simplify]], and [[carts-review]].

## Hard Rule

- Stage only files intended for this change.
- Do not revert unrelated user or parallel-agent edits.
- Run `dekk carts format` before commit unless the change is docs-only.
- Run [[carts-simplify]] then [[carts-review]] before committing.
- Once both gates pass with no outstanding findings, commit autonomously;
  do not pause for a second user confirmation. The gates *are* the gate.
- Push only when explicitly requested.

## Procedure

1. Inspect `git status --short` and separate intended from unrelated changes.
2. Run `dekk carts format` for code changes.
3. Run the verification command matched to the touched surface.
4. Invoke [[carts-simplify]] and address any simplification.
5. Invoke [[carts-review]] and address any findings.
6. Stage only intended files with explicit paths.
7. Write a 1-2 sentence commit message with a concrete CARTS scope.
8. Commit non-interactively. Push only if requested.

## Required Answer

List staged files, verification run, commit hash if created, and any unrelated
worktree changes left untouched.

## Autocommit Authorization

Once [[carts-simplify]] and [[carts-review]] have both run on the staged change
with no outstanding findings, run `git commit` without asking. Do not treat
the commit itself as a separate confirmation step.

Scope: any commit under `~/projects/carts/` (carts compiler + carts-plugin),
`~/projects/carts/external/carts-benchmarks/`, or `~/projects/carts-paper/`.

Edge cases:
- If [[carts-simplify]] flags a finding you cannot address without changing
  behavior, stop and surface it to the user. Do not commit through.
- For paper-only work in `~/projects/carts-paper/`, the analogous gates are
  the `paper-write-style` and `paper-write-revision` grep sets from the
  user-level rafa config. Same rule: pass the gates, then commit.
- `git push` remains explicitly gated by the Hard Rule above; this
  authorization is for `git commit` only.
- If the user says "don't commit yet" or "let me review first" for a
  specific change, that override beats this default for that change.
