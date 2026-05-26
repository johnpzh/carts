// RUN: %carts-compile %s --pass-pipeline='builtin.module()' | %FileCheck %s

// Cross-cutting round-trip for the `#arts.storage_bridge` ODS attribute.
// Pins the single-case `host_whole_to_compute_block` form on `arts.db_alloc`
// so the assembly format and ODS attribute name stay aligned.

// CHECK-LABEL: func.func @storage_bridge_attr_roundtrip
// CHECK: arts.db_alloc
// CHECK-SAME: storage_bridge = #arts.storage_bridge<host_whole_to_compute_block>

module {
  func.func @storage_bridge_attr_roundtrip() {
    %c1 = arith.constant 1 : index
    %route = arith.constant -1 : i32
    %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>, <coarse>]
        route(%route : i32) sizes[%c1] elementType(i32) elementSizes[%c1]
        {storage_bridge = #arts.storage_bridge<host_whole_to_compute_block>}
        : (memref<?xi64>, memref<?xmemref<?xi32>>)
    return
  }
}
