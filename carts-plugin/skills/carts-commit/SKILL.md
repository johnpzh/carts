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
