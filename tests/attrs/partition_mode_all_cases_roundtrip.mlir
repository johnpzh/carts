// RUN: %carts-compile %s --pass-pipeline='builtin.module()' | %FileCheck %s

// Cross-cutting round-trip for every `#arts.partition_mode` enum case.
// Pins all four ODS-declared cases (coarse, block, fine_grained, stencil) on
// `arts.db_alloc` so adding or renaming a case forces an explicit fixture
// update.

// CHECK-LABEL: func.func @partition_mode_all_cases_roundtrip
// CHECK: arts.db_alloc[<inout>, <heap>, <write>, <coarse>]
// CHECK: arts.db_alloc[<inout>, <heap>, <write>, <block>]
// CHECK: arts.db_alloc[<inout>, <heap>, <write>, <fine_grained>]
// CHECK: arts.db_alloc[<inout>, <heap>, <write>, <stencil>]

module {
  func.func @partition_mode_all_cases_roundtrip() {
    %c1 = arith.constant 1 : index
    %route = arith.constant -1 : i32
    %g0, %p0 = arts.db_alloc[<inout>, <heap>, <write>, <coarse>]
        route(%route : i32) sizes[%c1] elementType(i32) elementSizes[%c1]
        : (memref<?xi64>, memref<?xmemref<?xi32>>)
    %g1, %p1 = arts.db_alloc[<inout>, <heap>, <write>, <block>]
        route(%route : i32) sizes[%c1] elementType(i32) elementSizes[%c1]
        : (memref<?xi64>, memref<?xmemref<?xi32>>)
    %g2, %p2 = arts.db_alloc[<inout>, <heap>, <write>, <fine_grained>]
        route(%route : i32) sizes[%c1] elementType(i32) elementSizes[%c1]
        : (memref<?xi64>, memref<?xmemref<?xi32>>)
    %g3, %p3 = arts.db_alloc[<inout>, <heap>, <write>, <stencil>]
        route(%route : i32) sizes[%c1] elementType(i32) elementSizes[%c1]
        : (memref<?xi64>, memref<?xmemref<?xi32>>)
    return
  }
}
