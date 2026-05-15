// RUN: %carts-compile %s --pipeline post-db-refinement --start-from post-db-refinement --arts-config %inputs_dir/arts_1t.cfg | %FileCheck %s

// Verify that the late DbModeTightening pass preserves write-capable output
// modes after post-db-refinement. A full overwrite can tighten to `<out>`, but
// a partial overwrite must stay `<inout>` so untouched elements are preserved.
// Neither output dependency may degrade to `<in>`.

// CHECK: arts.db_alloc[<inout>, <heap>, <write>, <coarse>
// CHECK: arts.db_alloc[<inout>, <heap>, <write>, <coarse>
// CHECK: arts.db_acquire[<out>]
// CHECK: arts.db_acquire[<inout>]
// CHECK-NOT: arts.db_acquire[<in>]

module {
  func.func @preserve_write_output_modes() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %route = arith.constant -1 : i32
    %value = arith.constant 1.0 : f64

    %full_guid, %full_ptr = arts.db_alloc[<inout>, <heap>, <write>, <coarse>, <uniform>] route(%route : i32) sizes[%c1] elementType(f64) elementSizes[%c1] : (memref<?xi64>, memref<?xmemref<?xf64>>)
    %partial_guid, %partial_ptr = arts.db_alloc[<inout>, <heap>, <write>, <coarse>, <uniform>] route(%route : i32) sizes[%c1] elementType(f64) elementSizes[%c4] : (memref<?xi64>, memref<?xmemref<?xf64>>)

    %full_acq_guid, %full_acq_ptr = arts.db_acquire[<inout>] (%full_guid : memref<?xi64>, %full_ptr : memref<?xmemref<?xf64>>) partitioning(<coarse>), indices[], offsets[%c0], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?xf64>>)
    %partial_acq_guid, %partial_acq_ptr = arts.db_acquire[<inout>] (%partial_guid : memref<?xi64>, %partial_ptr : memref<?xmemref<?xf64>>) partitioning(<coarse>), indices[], offsets[%c0], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?xf64>>)

    arts.edt <task> <intranode> route(%route) (%full_acq_ptr, %partial_acq_ptr) : memref<?xmemref<?xf64>>, memref<?xmemref<?xf64>> {
    ^bb0(%full_dep: memref<?xmemref<?xf64>>, %partial_dep: memref<?xmemref<?xf64>>):
      %full = arts.db_ref %full_dep[%c0] : memref<?xmemref<?xf64>> -> memref<?xf64>
      memref.store %value, %full[%c0] : memref<?xf64>

      %partial = arts.db_ref %partial_dep[%c0] : memref<?xmemref<?xf64>> -> memref<?xf64>
      memref.store %value, %partial[%c0] : memref<?xf64>

      arts.db_release(%full_dep) : memref<?xmemref<?xf64>>
      arts.db_release(%partial_dep) : memref<?xmemref<?xf64>>
      arts.yield
    }

    arts.db_release(%full_acq_ptr) : memref<?xmemref<?xf64>>
    arts.db_release(%partial_acq_ptr) : memref<?xmemref<?xf64>>
    return
  }
}
