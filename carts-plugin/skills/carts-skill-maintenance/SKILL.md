---
name: carts-skill-maintenance
description: Use when creating, regenerating, validating, or hardening CARTS project skills and agent resources.
---

# CARTS Skill Maintenance

Skills are process contracts. Descriptions should say when to load a skill, not
summarize the whole workflow.

## Procedure To Add A New Skill

1. Create `carts-plugin/skills/<name>/SKILL.md` with frontmatter, trigger,
   Hard Rule, Procedure, and Required Answer or exit criteria.
2. Keep body text under 150 lines; move bulky compiler facts to `references/`.
3. Use one trigger-only description beginning with `Use when`.
4. Cross-link sibling skills with `[[skill-name]]` instead of duplicating.
5. Keep generated diagnostics, dumps, and scratch sessions under
   `.carts/outputs` or `.carts/sessions`; do not put `/tmp` paths in skill
   examples unless a tool requires it.
6. Check governance before generation:
   - skill directory names should match frontmatter `name` for new skills;
   - descriptions start with `Use when` and describe triggers, not workflow;
   - examples use `dekk carts ...`, not raw build tools;
   - `carts-simplify` is referenced from pass, review, and maintenance flows;
   - source plugin files remain agent-agnostic; do not add host-specific
     environment variables to `carts-plugin/hooks`, MCP tools, or source
     skills; use only project-owned `CARTS_*` roots or repository discovery;
   - plugin metadata stays present for supported agent targets.
7. Generate resources:
   ```bash
   dekk carts skills generate
   ```
8. Check discovery:
   ```bash
   dekk carts skills list
   dekk carts skills status
   ```
9. Check generated adapters for source-policy drift:
   ```bash
   rg --no-ignore 'PROJECT_DIR|PLUGIN_ROOT|python tools/carts_cli.py' \
     carts-plugin/hooks carts-plugin/mcp carts-plugin/.mcp.json
   ```
   This should show only `CARTS_PROJECT_DIR`, `CARTS_PLUGIN_ROOT`, and
   repository discovery. If it finds host-specific roots or raw implementation
   commands, patch the generated adapter back to project-owned roots,
   `git rev-parse`, and `dekk carts ...` before commit.
10. If discovery does not use the configured `[agents].source`, fix the tooling
   before relying on generated resources.

## Hard Rule

- Keep `carts-plugin/` as the repo-managed source of truth.
- Descriptions are trigger sentences, not workflow summaries.
- Keep `SKILL.md` bodies under 150 lines; move bulky facts to references.
- Cross-link siblings with `[[skill-name]]` instead of duplicating.
- Use `dekk carts ...` commands in examples.

## Required Answer

List created or edited skill files, generated-resource commands, discovery
checks, and any source-policy drift found.

## Pressure Tests

Before calling skills hardened, test prompts that would otherwise cause agents
to use stale pipeline docs, use bare `carts` without checking, refresh fixtures
blindly, skip root-cause analysis, claim success without verification, add
duplicate pass-local helpers, dump artifacts outside `.carts/`, or mix behavior
changes with unrelated refactors.
