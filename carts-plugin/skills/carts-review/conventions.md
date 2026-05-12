# CARTS Compiler Conventions

## Attribute Names

Use centralized attribute names:

- `include/arts/utils/OperationAttributes.h`
- `include/arts/utils/StencilAttributes.h`

Do not hardcode project attribute strings in pass logic.

Quick scan:

```bash
rg -n '"arts\.' lib/arts --glob '*.cpp' | rg -v 'AttrNames|DEBUG|ARTS_DEBUG|arts_debug'
```

## Analysis Access

Use `AnalysisManager` interfaces for DB, EDT, loop, metadata, and cache state.
Do not bypass them by directly reaching into graph internals unless you are
inside the analysis implementation itself.

Quick scan:

```bash
rg -n 'getDbGraph|getEdtGraph|\.getGraph\(\)|\.invalidate\(' lib/arts --glob '*.cpp'
```

## Utility Duplication

Before adding any static helper, check if it exists in shared utils. Use
`carts-check-utils` for helpers, predicates, string utilities, IR queries, and
attribute helpers.

Quick scan:

```bash
rg -n '^static .*\\(' lib/arts include/arts --glob '*.cpp' --glob '*.h'
```

## File Placement

- SDE semantics: `lib/arts/dialect/sde`.
- Core DB/EDT/epoch/analysis: `lib/arts/dialect/core`.
- Runtime-shaped lowering: `lib/arts/dialect/rt`.
- Shared utilities: `include/arts/utils`, `lib/arts/utils`, or dialect-specific
  support utilities when the helper is not globally meaningful.

## Review Traps

- Stale docs naming non-live stages or passes.
- Fixture refresh hiding a real verifier failure.
- New static helpers duplicating existing utilities.
- Runtime debug build left as final verification.
- Examples runner mutating sample artifacts during a supposedly read-only task.
