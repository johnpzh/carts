// RUN: %carts-compile %s --arts-config %arts_config --start-from initial-cleanup --pipeline initial-cleanup | %FileCheck %s

// CHECK-LABEL: func.func @explicit_edt_param_accepts_stack_alloca_capture
// CHECK: arts.edt <task> <intranode>
// CHECK: memref.store %{{.*}}, %[[SCRATCH:.*]][] : memref<i32>
// CHECK: memref.load %[[SCRATCH]][] : memref<i32>

module attributes {arts.runtime_total_nodes = 1 : i64, arts.runtime_total_workers = 8 : i64} {
  func.func @explicit_edt_param_accepts_stack_alloca_capture() {
    %route = arith.constant -1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %v = arith.constant 7 : i32
    %scratch = memref.alloca() : memref<i32>

    %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>, <coarse>] route(%route : i32) sizes[%c1] elementType(i32) elementSizes[%c1] : (memref<?xi64>, memref<?xmemref<?xi32>>)
    %acq_guid, %acq_ptr = arts.db_acquire[<inout>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<?xi32>>) partitioning(<coarse>), indices[], offsets[%c0], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?xi32>>)

    arts.edt <task> <intranode> route(%route) (%acq_ptr) : memref<?xmemref<?xi32>> params(%c1 : index) {
    ^bb0(%dep: memref<?xmemref<?xi32>>, %p_arg: index):
      memref.store %v, %scratch[] : memref<i32>
      %local = memref.load %scratch[] : memref<i32>
      %payload = arts.db_ref %dep[%c0] : memref<?xmemref<?xi32>> -> memref<?xi32>
      memref.store %local, %payload[%c0] : memref<?xi32>
      arts.yield
    }

    arts.db_release(%acq_ptr) : memref<?xmemref<?xi32>>
    return
  }
}
