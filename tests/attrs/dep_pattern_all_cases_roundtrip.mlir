// RUN: %carts-compile %s --pass-pipeline='builtin.module()' | %FileCheck %s

// Cross-cutting round-trip for every `#arts.dep_pattern` enum case carried as
// the `depPattern` discardable attribute on `arts.edt`. Pins all twelve cases
// (unknown plus eleven canonical families) so any rename, reorder, or removal
// breaks here instead of silently round-tripping under a stale alias.

// CHECK-LABEL: func.func @dep_pattern_all_cases_roundtrip
// CHECK: depPattern = #arts.dep_pattern<unknown>
// CHECK: depPattern = #arts.dep_pattern<uniform>
// CHECK: depPattern = #arts.dep_pattern<stencil>
// CHECK: depPattern = #arts.dep_pattern<matmul>
// CHECK: depPattern = #arts.dep_pattern<triangular>
// CHECK: depPattern = #arts.dep_pattern<wavefront_2d>
// CHECK: depPattern = #arts.dep_pattern<jacobi_alternating_buffers>
// CHECK: depPattern = #arts.dep_pattern<elementwise_pipeline>
// CHECK: depPattern = #arts.dep_pattern<stencil_tiling_nd>
// CHECK: depPattern = #arts.dep_pattern<cross_dim_stencil_3d>
// CHECK: depPattern = #arts.dep_pattern<higher_order_stencil>
// CHECK: depPattern = #arts.dep_pattern<reduction>

module {
  func.func @dep_pattern_all_cases_roundtrip() {
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
        attributes {depPattern = #arts.dep_pattern<unknown>} {
    ^bb0(%a: memref<?xmemref<?xi32>>):
      arts.yield
    }
    arts.edt <task> <intranode> route(%route) (%ap) : memref<?xmemref<?xi32>>
        attributes {depPattern = #arts.dep_pattern<uniform>} {
    ^bb0(%a: memref<?xmemref<?xi32>>):
      arts.yield
    }
    arts.edt <task> <intranode> route(%route) (%ap) : memref<?xmemref<?xi32>>
        attributes {depPattern = #arts.dep_pattern<stencil>} {
    ^bb0(%a: memref<?xmemref<?xi32>>):
      arts.yield
    }
    arts.edt <task> <intranode> route(%route) (%ap) : memref<?xmemref<?xi32>>
        attributes {depPattern = #arts.dep_pattern<matmul>} {
    ^bb0(%a: memref<?xmemref<?xi32>>):
      arts.yield
    }
    arts.edt <task> <intranode> route(%route) (%ap) : memref<?xmemref<?xi32>>
        attributes {depPattern = #arts.dep_pattern<triangular>} {
    ^bb0(%a: memref<?xmemref<?xi32>>):
      arts.yield
    }
    arts.edt <task> <intranode> route(%route) (%ap) : memref<?xmemref<?xi32>>
        attributes {depPattern = #arts.dep_pattern<wavefront_2d>} {
    ^bb0(%a: memref<?xmemref<?xi32>>):
      arts.yield
    }
    arts.edt <task> <intranode> route(%route) (%ap) : memref<?xmemref<?xi32>>
        attributes {depPattern = #arts.dep_pattern<jacobi_alternating_buffers>} {
    ^bb0(%a: memref<?xmemref<?xi32>>):
      arts.yield
    }
    arts.edt <task> <intranode> route(%route) (%ap) : memref<?xmemref<?xi32>>
        attributes {depPattern = #arts.dep_pattern<elementwise_pipeline>} {
    ^bb0(%a: memref<?xmemref<?xi32>>):
      arts.yield
    }
    arts.edt <task> <intranode> route(%route) (%ap) : memref<?xmemref<?xi32>>
        attributes {depPattern = #arts.dep_pattern<stencil_tiling_nd>} {
    ^bb0(%a: memref<?xmemref<?xi32>>):
      arts.yield
    }
    arts.edt <task> <intranode> route(%route) (%ap) : memref<?xmemref<?xi32>>
        attributes {depPattern = #arts.dep_pattern<cross_dim_stencil_3d>} {
    ^bb0(%a: memref<?xmemref<?xi32>>):
      arts.yield
    }
    arts.edt <task> <intranode> route(%route) (%ap) : memref<?xmemref<?xi32>>
        attributes {depPattern = #arts.dep_pattern<higher_order_stencil>} {
    ^bb0(%a: memref<?xmemref<?xi32>>):
      arts.yield
    }
    arts.edt <task> <intranode> route(%route) (%ap) : memref<?xmemref<?xi32>>
        attributes {depPattern = #arts.dep_pattern<reduction>} {
    ^bb0(%a: memref<?xmemref<?xi32>>):
      arts.yield
    }
    return
  }
}
