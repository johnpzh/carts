// RUN: not %carts-compile %s --arts-config %arts_config --start-from initial-cleanup --pipeline initial-cleanup 2>&1 | %FileCheck %s

// CHECK: EDT region captures scalar value
// CHECK-SAME: pass it as an EDT param and use the param block argument

module attributes {arts.runtime_total_nodes = 1 : i64, arts.runtime_total_workers = 8 : i64} {
  func.func private @sink(index)

  func.func @implicit_scalar_capture_is_rejected(%captured: index) {
    %route = arith.constant -1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>, <coarse>] route(%route : i32) sizes[%c1] elementType(i32) elementSizes[%c1] : (memref<?xi64>, memref<?xmemref<?xi32>>)
    %acq_guid, %acq_ptr = arts.db_acquire[<inout>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<?xi32>>) partitioning(<coarse>), indices[], offsets[%c0], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?xi32>>)

    arts.edt <task> <intranode> route(%route) (%acq_ptr) : memref<?xmemref<?xi32>> {
    ^bb0(%dep: memref<?xmemref<?xi32>>):
      func.call @sink(%captured) : (index) -> ()
      arts.yield
    }

    return
  }
}
