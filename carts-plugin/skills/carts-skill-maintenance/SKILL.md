---
name: carts-skill-maintenance
description: Use when creating, regenerating, validating, or hardening CARTS project skills, agent resources, trigger descriptions, skill references, or dekk carts skills behavior.
---

# CARTS Skill Maintenance

Skills are process contracts. Descriptions should say when to load a skill, not
summarize the whole workflow.

## Workflow

1. Keep `carts-plugin/` as the repo-managed source of truth unless `.dekk.toml`
   changes.
2. Put concise procedures in `SKILL.md`; put bulky compiler facts in
   `references/`.
3. Use trigger-only descriptions beginning with `Use when`.
4. Keep generated diagnostics, dumps, and scratch sessions under
   `.carts/outputs` or `.carts/sessions`; do not put `/tmp` paths in skill
   examples unless a tool requires it.
5. Check governance before generation:
   - skill directory names should match frontmatter `name` for new skills;
   - descriptions start with `Use when` and describe triggers, not workflow;
   - examples use `dekk carts ...`, not raw build tools;
   - `carts-simplify` is referenced from pass, review, and maintenance flows;
   - source plugin files remain agent-agnostic; do not add host-specific
     environment variables to `carts-plugin/hooks`, MCP tools, or source
     skills; use only project-owned `CARTS_*` roots or repository discovery;
   - plugin metadata stays present for supported agent targets.
6. Generate resources:
   ```bash
   dekk carts skills generate
   ```
7. Check discovery:
   ```bash
   dekk carts skills list
   dekk carts skills status
   ```
8. Check generated adapters for source-policy drift:
   ```bash
   rg --no-ignore 'PROJECT_DIR|PLUGIN_ROOT|python tools/carts_cli.py' \
     carts-plugin/hooks carts-plugin/mcp carts-plugin/.mcp.json
   ```
   This should show only `CARTS_PROJECT_DIR`, `CARTS_PLUGIN_ROOT`, and
   repository discovery. If it finds host-specific roots or raw implementation
   commands, patch the generated adapter back to project-owned roots,
   `git rev-parse`, and `dekk carts ...` before commit.
9. If discovery does not use the configured `[agents].source`, fix the tooling
   before relying on generated resources.

## Pressure Tests

Before calling skills hardened, test prompts that would otherwise cause agents
to use stale pipeline docs, use bare `carts` without checking, refresh fixtures
blindly, skip root-cause analysis, claim success without verification, add
duplicate pass-local helpers, dump artifacts outside `.carts/`, or mix behavior
changes with unrelated refactors.
