// RUN: %carts-compile %s --pass-pipeline='builtin.module(arts-dce)' | %FileCheck %s

module {
  func.func private @use_ptr(memref<?xmemref<?xf64>>)

  func.func @removes_dead_arts_objects() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %route = arith.constant -1 : i32
    %dead = arts.undef : i32
    %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>, <coarse>] route(%route : i32) sizes[%c1] elementType(f64) elementSizes[%c1] : (memref<?xi64>, memref<?xmemref<?xf64>>)
    %acq_guid, %acq_ptr = arts.db_acquire[<in>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<?xf64>>) partitioning(<coarse>), indices[], offsets[%c0], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?xf64>>)
    arts.db_release(%acq_ptr) : memref<?xmemref<?xf64>>
    arts.db_free(%guid) : memref<?xi64>
    arts.db_free(%ptr) : memref<?xmemref<?xf64>>
    arts.edt <task> <intranode> route(%route) {
      arts.yield
    }
    return
  }

  func.func @keeps_live_acquire_and_drops_duplicate_release() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %route = arith.constant -1 : i32
    %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>, <coarse>] route(%route : i32) sizes[%c1] elementType(f64) elementSizes[%c1] : (memref<?xi64>, memref<?xmemref<?xf64>>)
    %acq_guid, %acq_ptr = arts.db_acquire[<in>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<?xf64>>) partitioning(<coarse>), indices[], offsets[%c0], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?xf64>>)
    func.call @use_ptr(%acq_ptr) : (memref<?xmemref<?xf64>>) -> ()
    arts.db_release(%acq_ptr) : memref<?xmemref<?xf64>>
    arts.db_release(%acq_ptr) : memref<?xmemref<?xf64>>
    return
  }
}

// CHECK-LABEL: func.func @removes_dead_arts_objects
// CHECK-NOT: arts.undef
// CHECK-NOT: arts.db_alloc
// CHECK-NOT: arts.db_acquire
// CHECK-NOT: arts.db_release
// CHECK-NOT: arts.db_free
// CHECK-NOT: arts.edt
// CHECK: return

// CHECK-LABEL: func.func @keeps_live_acquire_and_drops_duplicate_release
// CHECK: arts.db_alloc
// CHECK: %[[ACQ_GUID:.*]], %[[ACQ_PTR:.*]] = arts.db_acquire
// CHECK: call @use_ptr(%[[ACQ_PTR]])
// CHECK: arts.db_release(%[[ACQ_PTR]])
// CHECK-NOT: arts.db_release(%[[ACQ_PTR]])
// CHECK: return
