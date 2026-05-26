// RUN: %carts-compile %s --pass-pipeline='builtin.module()' | %FileCheck %s

// Cross-cutting round-trip for every `#arts.distribution_kind` enum case
// carried as the `distribution_kind` discardable attribute on `arts.edt`.
// Pins all five cases (block, two_level, block_cyclic, tiling_2d, replicated)
// so adding a strategy forces an explicit fixture update.

// CHECK-LABEL: func.func @distribution_kind_all_cases_roundtrip
// CHECK: distribution_kind = #arts.distribution_kind<block>
// CHECK: distribution_kind = #arts.distribution_kind<two_level>
// CHECK: distribution_kind = #arts.distribution_kind<block_cyclic>
// CHECK: distribution_kind = #arts.distribution_kind<tiling_2d>
// CHECK: distribution_kind = #arts.distribution_kind<replicated>

module {
  func.func @distribution_kind_all_cases_roundtrip() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %route = arith.constant -1 : i32
    %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>, <coarse>]
        route(%route : i32) sizes[%c1] elementType(i32) elementSizes[%c1]
        : (memref<?xi64>, memref<?xmemref<?xi32>>)
    %ag, %ap = arts.db_acquire[<inout>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<?xi32>>)
        partitioning(<coarse>), indices[], offsets[%c0], sizes[%c1]
        -> (memref<?xi64>, memref<?xmemref<?xi32>>)
    arts.edt <task> <intranode> route(%route) (%ap) : memref<?xmemref<?xi32>>
        attributes {distribution_kind = #arts.distribution_kind<block>} {
    ^bb0(%a: memref<?xmemref<?xi32>>):
      arts.yield
    }
    arts.edt <task> <intranode> route(%route) (%ap) : memref<?xmemref<?xi32>>
        attributes {distribution_kind = #arts.distribution_kind<two_level>} {
    ^bb0(%a: memref<?xmemref<?xi32>>):
      arts.yield
    }
    arts.edt <task> <intranode> route(%route) (%ap) : memref<?xmemref<?xi32>>
        attributes {distribution_kind = #arts.distribution_kind<block_cyclic>} {
    ^bb0(%a: memref<?xmemref<?xi32>>):
      arts.yield
    }
    arts.edt <task> <intranode> route(%route) (%ap) : memref<?xmemref<?xi32>>
        attributes {distribution_kind = #arts.distribution_kind<tiling_2d>} {
    ^bb0(%a: memref<?xmemref<?xi32>>):
      arts.yield
    }
    arts.edt <task> <intranode> route(%route) (%ap) : memref<?xmemref<?xi32>>
        attributes {distribution_kind = #arts.distribution_kind<replicated>} {
    ^bb0(%a: memref<?xmemref<?xi32>>):
      arts.yield
    }
    return
  }
}
