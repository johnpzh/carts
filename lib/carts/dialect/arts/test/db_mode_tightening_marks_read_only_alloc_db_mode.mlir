// RUN: %carts-compile %s --pipeline post-db-refinement --start-from post-db-refinement --arts-config %inputs_dir/arts_1t.cfg | %FileCheck %s

// A DB whose visible acquires are read-only must tighten both the ARTS access
// mode and the runtime DB mode. ARTS-RT uses the DB mode when deciding whether
// distributed read dependencies may request duplicate-frontier reads.

// CHECK-LABEL: func.func @read_only_alloc_db_mode
// CHECK: arts.db_alloc[<in>, <heap>, <read>, <coarse>
// CHECK: arts.db_acquire[<in>]

module {
  func.func private @sink(f64)

  func.func @read_only_alloc_db_mode() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %route = arith.constant -1 : i32

    %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>, <coarse>, <uniform>] route(%route : i32) sizes[%c1] elementType(f64) elementSizes[%c1] : (memref<?xi64>, memref<?xmemref<?xf64>>)
    %acq_guid, %acq_ptr = arts.db_acquire[<inout>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<?xf64>>) partitioning(<coarse>), indices[], offsets[%c0], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?xf64>>)

    arts.edt <task> <intranode> route(%route) (%acq_ptr) : memref<?xmemref<?xf64>> {
    ^bb0(%dep: memref<?xmemref<?xf64>>):
      %payload = arts.db_ref %dep[%c0] : memref<?xmemref<?xf64>> -> memref<?xf64>
      %value = memref.load %payload[%c0] : memref<?xf64>
      func.call @sink(%value) : (f64) -> ()
      arts.db_release(%dep) : memref<?xmemref<?xf64>>
      arts.yield
    }

    arts.db_release(%acq_ptr) : memref<?xmemref<?xf64>>
    return
  }
}
