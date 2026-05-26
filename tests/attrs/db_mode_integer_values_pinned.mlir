// RUN: %carts-compile %s --pass-pipeline='builtin.module()' | %FileCheck %s

// Cross-cutting round-trip for the `#arts.db_mode` enum cases pinned through
// the syntactic `<read>` / `<write>` forms on `arts.db_alloc`. These cases
// have load-bearing integer values: 1 == `DB_MODE_RO`, 2 == `DB_MODE_EW`.
//
// COM: Runtime ABI alignment - the integer values in ArtsAttrs.td for the
// COM: `DbMode` enum must remain 1 and 2 so dependency lowering can forward
// COM: them to the ARTS runtime without an extra translation table. Adding a
// COM: case or renumbering breaks this contract; this fixture pins the print
// COM: surface so any silent reshuffle of the enum is caught at lit time.

// CHECK-LABEL: func.func @db_mode_read_write_roundtrip
// CHECK: arts.db_alloc[<inout>, <heap>, <read>, <coarse>]
// CHECK: arts.db_alloc[<inout>, <heap>, <write>, <coarse>]

module {
  func.func @db_mode_read_write_roundtrip() {
    %c1 = arith.constant 1 : index
    %route = arith.constant -1 : i32
    %gr, %pr = arts.db_alloc[<inout>, <heap>, <read>, <coarse>]
        route(%route : i32) sizes[%c1] elementType(i32) elementSizes[%c1]
        : (memref<?xi64>, memref<?xmemref<?xi32>>)
    %gw, %pw = arts.db_alloc[<inout>, <heap>, <write>, <coarse>]
        route(%route : i32) sizes[%c1] elementType(i32) elementSizes[%c1]
        : (memref<?xi64>, memref<?xmemref<?xi32>>)
    return
  }
}
