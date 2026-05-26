---
name: carts-attr-consolidation
description: Use when adding or changing CARTS TableGen attrs, enum attrs, raw dialect attr strings, convert* enum switches, or AttrNames.
---

# CARTS Attribute Consolidation

Use with [[carts-dialect-map]] for boundary ownership and [[carts-check-utils]]
when attribute access is hidden inside a helper.

## Hard Rule

- Every CARTS IR attribute is ODS-declared in the owning dialect before C++ use.
- Raw `arts.*`, `sde.*`, `codir.*`, and `arts_rt.*` attr API strings are banned.
- New `AttrNames::` entries for CARTS IR are blocked.
- Identical enum case sets across dialects are hoisted, not converted.
- Tier-2 enums need semantic alignment before consolidation.

## Attribute Snapshot

| Tier | Enum | Dialects and cases |
|------|------|--------------------|
| 1 hoist | `IterationTopology` | SDE/CODIR/ARTS, 3 identical cases |
| 1 hoist | `ReductionStrategy` | SDE/CODIR/ARTS, 3 identical cases |
| 1 hoist | `StorageViewKind` | SDE/CODIR, 4 identical cases |
| 1 hoist | `BarrierReason` | SDE/ARTS, 5 identical cases |
| 2 defer | `AccessMode` | SDE 3, ARTS 4, ARTS-RT 2 cases |
| 2 defer | `Pattern`/`DepPattern` | SDE/CODIR 9, ARTS 12 cases |
| 2 defer | `DistributionKind` | SDE/CODIR 3, ARTS 5 cases |
| 2 defer | `RepetitionStructure` | SDE/CODIR 4, ARTS 2 cases |
| 2 defer | `AsyncStrategy` | SDE/CODIR 3, ARTS 2 cases |

## Procedure

1. Identify the producer op or op set and the semantic owning dialect.
2. Grep sibling dialect `.td` files for the same enum or identical case set.
3. If the case set is Tier 1 or otherwise identical, hoist to
   `include/carts/IR/CommonAttrs.td` and delete rename-only conversions.
4. If the enum is Tier 2, do not merge it in the same patch; document the
   semantic mismatch.
5. Declare the attribute in the owning dialect `*Attrs.td`/`*Ops.td`.
6. Replace C++ access with generated `op.get<Name>AttrName()`, `op.get<Name>()`,
   or `op.set<Name>(...)`.
7. For multi-op attributes, use a free-standing ODS attr plus trait/interface.
8. Delete the raw string or `AttrNames::` entry in the same patch.

## Smell Detector

A `convert<Name>` switch with N-to-N identity mapping is evidence that the enum
belongs in a shared ODS definition. Do not preserve a pure rename conversion.

## Detection Greps

```bash
rg 'getAttr\("(arts|sde|codir|arts_rt)\.[^"]*"\)' include/ lib/
rg 'setAttr\("(arts|sde|codir|arts_rt)\.[^"]*"' include/ lib/
rg 'hasAttr\("(arts|sde|codir|arts_rt)\.[^"]*"\)' include/ lib/
rg 'removeAttr\("(arts|sde|codir|arts_rt)\.[^"]*"\)' include/ lib/
rg 'StringAttr::get\([^,]+,\s*"(arts|sde|codir|arts_rt)\.' include/ lib/
rg '(getDiscardableAttr|setDiscardableAttr|getInherentAttr|setInherentAttr)\("(arts|sde|codir|arts_rt)\.' include/ lib/
rg 'getAttrOfType<[^>]+>\("(arts|sde|codir|arts_rt)\.' include/ lib/
rg 'AttrNames::[A-Za-z:]+' include/ lib/
```

## Required Answer

State the owning op or op set, owning dialect, Tier 1/Tier 2 status, ODS file,
accessor replacement, and deletion target for every migrated attribute.
