---
name: carts-agentic-development
description: Use when planning or executing CARTS work with multiple agents, independent tasks, implementation plans, code review checkpoints, or staged compiler/runtime investigations.
---

# CARTS Agentic Development

Use isolated agents for bounded, independent work; keep orchestration and final
integration local.

## Process

1. Build a concrete task list with files, stages, expected tests, and ownership.
2. Give each worker a disjoint write set or a read-only question.
3. Do not let parallel workers edit the same files.
4. Review in two stages:
   - spec compliance: did it do exactly the requested CARTS task?
   - CARTS quality: dialect placement, attributes, analyses, tests,
     verification.
5. Integrate only after fresh local verification.

## CARTS-Specific Review Gate

Check `carts-simplify`, `carts-review`, `carts-test`, and the relevant domain
skill before marking task completion. Integration owns the final simplification
pass even when workers already simplified their own patches.
