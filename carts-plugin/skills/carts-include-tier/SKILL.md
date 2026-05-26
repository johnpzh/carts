---
name: carts-include-tier
description: Use when lib headers gain multiple consumers, adding lib *Utils.h, or finding include *Internal.h.
---

# CARTS Include Tier

Use with [[carts-check-utils]] for helper ownership and [[carts-refactor-utils]]
for the actual extraction patch.

## Hard Rule

- `lib/` headers are pass-private implementation details.
- Multi-consumer utility headers belong in the parallel `include/carts/...` path.
- `*Internal.h` belongs under `lib/` when it serves one pass family.
- Cross-dialect public contracts need an owning public include path.
- Do not expose a header just to avoid deleting duplicated pass-local helpers.

## Procedure

1. Count consumers with `rg '#include ".*<HeaderName>"' include/carts lib/carts`.
2. Find lib-tier utility headers and users:
   ```bash
   rg --files-with-matches '#include ".*Utils\.h"' lib/carts/
   ```
3. Find misplaced public internals:
   ```bash
   find include/carts -name '*Internal.h'
   ```
4. Classify the header with the decision table.
5. Move declarations and includes in the same patch when the tier changes.
6. Run `dekk carts format` and the smallest build/test covering the consumers.

## Decision Table

| Consumers | Pass-private? | Cross-dialect? | Correct tier |
|-----------|---------------|----------------|--------------|
| one `.cpp` | yes | no | `lib/carts/.../<PassName>Internal.h` or pass-local |
| one pass family | yes | no | `lib/carts/.../Transforms/<area>/` |
| multiple `.cpp` | no | no | parallel `include/carts/<dialect-or-utils>/...` |
| multiple dialects | no | yes | public owner in `include/carts/...` |
| generated or ODS contract | no | maybe | owning dialect `include/carts/.../IR` |

## Required Answer

State `Keep under lib`, `Promote to include`, or `Move down to lib`, with the
consumer count, privacy reason, and target path.
