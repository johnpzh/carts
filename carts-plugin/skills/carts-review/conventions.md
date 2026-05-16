# CARTS Compiler Conventions

## Attribute Names

Use centralized attribute names:

- `include/carts/utils/OperationAttributes.h`
- `include/carts/utils/StencilAttributes.h`

Do not hardcode project attribute strings in pass logic.

Quick scan:

```bash
rg -n '"arts\.' lib/carts include/carts --glob '*.cpp' --glob '*.h' | rg -v 'AttrNames|DEBUG|ARTS_DEBUG|arts_debug'
```

## Analysis Access

Use `AnalysisManager` interfaces for DB, EDT, loop, metadata, and cache state.
Do not bypass them by directly reaching into graph internals unless you are
inside the analysis implementation itself.

Quick scan:

```bash
rg -n 'getDbGraph|getEdtGraph|\.getGraph\(\)|\.invalidate\(' lib/carts include/carts --glob '*.cpp' --glob '*.h'
```

## Utility Duplication

Before adding any static helper, check if it exists in shared utils. Use
`carts-check-utils` for helpers, predicates, string utilities, IR queries, and
attribute helpers.

Quick scan:

```bash
rg -n '^static .*\\(' lib/carts include/carts --glob '*.cpp' --glob '*.h'
```

## File Placement

- SDE semantics: `lib/carts/dialect/sde`.
- CODIR codelet isolation: `lib/carts/dialect/codir`.
- ARTS DB/EDT/epoch/analysis: `lib/carts/dialect/arts`.
- ARTS-RT runtime-shaped lowering: `lib/carts/dialect/arts-rt`.
- Shared utilities: `include/carts/utils`, `lib/carts/utils`, or dialect-specific
  support utilities when the helper is not globally meaningful.

## Review Traps

- Stale docs naming non-live stages or passes.
- Fixture refresh hiding a real verifier failure.
- New static helpers duplicating existing utilities.
- Runtime debug build left as final verification.
- Examples runner mutating sample artifacts during a supposedly read-only task.
