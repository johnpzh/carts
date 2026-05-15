// RUN: not %carts-compile %s --arts-config %arts_config --start-from initial-cleanup --pipeline initial-cleanup 2>&1 | %FileCheck %s

// CHECK: EDT region captures value
// CHECK-SAME: pass it as an EDT dependency or param and use the corresponding block argument

module attributes {arts.runtime_total_nodes = 1 : i64, arts.runtime_total_workers = 8 : i64} {
  func.func private @sink_vec(vector<4xi32>)

  func.func @explicit_edt_param_rejects_vector_capture(%captured: vector<4xi32>) {
    %route = arith.constant -1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %p = arith.constant 7 : index

    %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>, <coarse>] route(%route : i32) sizes[%c1] elementType(i32) elementSizes[%c1] : (memref<?xi64>, memref<?xmemref<?xi32>>)
    %acq_guid, %acq_ptr = arts.db_acquire[<inout>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<?xi32>>) partitioning(<coarse>), indices[], offsets[%c0], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?xi32>>)

    arts.edt <task> <intranode> route(%route) (%acq_ptr) : memref<?xmemref<?xi32>> params(%p : index) {
    ^bb0(%dep: memref<?xmemref<?xi32>>, %p_arg: index):
      func.call @sink_vec(%captured) : (vector<4xi32>) -> ()
      arts.yield
    }

    return
  }
}
