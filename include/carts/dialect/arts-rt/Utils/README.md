# ARTS-RT Utils

ARTS-RT utilities own helpers for the runtime ABI and lowering-ready IR shape.
Use this directory for runtime call construction, DB lowering provenance,
stable ID assignment for runtime objects, depv layout support, packing, pointer
lowering, and LLVM-facing cleanup helpers.

Do not put abstract DB/EDT scheduling, placement, distributed ownership, or
ARTS object analysis here. Those decisions belong in `arts/Utils` or
`arts/Analysis` before the runtime ABI is selected.

Current utility groups:

- `RuntimeCallUtils` constructs calls to the ARTS runtime ABI.
- `RtDbUtils` follows runtime-dialect DB provenance after ARTS objects have
  started lowering.
- `IdRegistry` assigns stable runtime object IDs for lowered runtime state.
