// RUN: %carts-compile %s --arts-config %arts_config --start-from late-concurrency-cleanup --pipeline late-concurrency-cleanup | %FileCheck %s

// Loop-invariant inout acquires that only feed EDTs inside an epoch can be
// acquired once around the epoch. The epoch remains the completion boundary,
// so the write-capable mode is preserved instead of paying one acquire per
// worker tile.

// CHECK-LABEL: func.func @hoist_inout_acquire_out_of_epoch
// CHECK: arts.db_acquire[<inout>]
// CHECK: arts.epoch
// CHECK-NOT: arts.db_acquire[<inout>]
// CHECK: arts.edt
// CHECK: arts.db_release

module {
  func.func @hoist_inout_acquire_out_of_epoch() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %route = arith.constant -1 : i32
    %value = arith.constant 1.0 : f64

    %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>, <coarse>, <uniform>] route(%route : i32) sizes[%c1] elementType(f64) elementSizes[%c4] : (memref<?xi64>, memref<?xmemref<?xf64>>)
    arts.epoch {
      scf.for %i = %c0 to %c4 step %c1 {
        %acq_guid, %acq_ptr = arts.db_acquire[<inout>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<?xf64>>) partitioning(<coarse>), indices[], offsets[%c0], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?xf64>>)
        arts.edt <task> <intranode> route(%route) (%acq_ptr) : memref<?xmemref<?xf64>> {
        ^bb0(%dep: memref<?xmemref<?xf64>>):
          %view = arts.db_ref %dep[%c0] : memref<?xmemref<?xf64>> -> memref<?xf64>
          memref.store %value, %view[%c0] : memref<?xf64>
          arts.yield
        }
      }
      arts.yield
    } : i64

    arts.db_free(%guid) : memref<?xi64>
    arts.db_free(%ptr) : memref<?xmemref<?xf64>>
    return
  }
}
