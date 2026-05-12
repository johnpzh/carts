# CARTS Agent Skills

`carts-plugin/` is the repo-managed source of truth for CARTS agent guidance.
It is intentionally centered on skills and compact references rather than
custom workflow engines.

Generate agent-facing resources with:

```bash
dekk carts skills generate
```

Useful checks:

```bash
dekk carts skills list
dekk carts skills status
dekk carts skills view carts-cli
```

The skill suite is organized around the compiler workflow:

- Foundation: command policy, live pipeline/dialect maps, utility checks, and
  final simplification.
- Build/test: compiler builds, lit/e2e verification, test creation, and review.
- Compiler development: pass placement, runtime-first lowering, and contract
  refreshes.
- Debug/triage: compiler failures, miscompiles, runtime failures, stage diffs,
  reproducers, stale analyses, and distributed behavior.
- Performance/distributed: benchmarks, benchmark triage, multinode examples,
  and heuristic explanations.
- Meta: multi-agent development and skill maintenance.

Generated resources may include `AGENTS.md`, `CLAUDE.md`, `.claude/skills/`,
`.cursorrules`, `.github/copilot-instructions.md`, and `.agents.json`,
depending on the active dekk version and configured targets.
