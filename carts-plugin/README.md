# CARTS Agent Skills

`carts-plugin/` is the repo-managed source of truth for CARTS agent guidance.
It is intentionally centered on skills and compact references rather than
custom workflow engines.

The source plugin is agent-agnostic. Host-specific instruction files, skill
exports, MCP configs, and editor integrations are adapters emitted from this
directory; do not put agent-specific environment variables or workflow
assumptions in source skills, hooks, or MCP tools. Use `CARTS_PROJECT_DIR` /
`CARTS_PLUGIN_ROOT` when a hook needs an explicit root, and otherwise resolve
the repository with `git`.

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

Generated resources may include root instruction files, skill exports, editor
configuration, MCP configuration, and agent registries depending on the active
dekk version and configured targets.
