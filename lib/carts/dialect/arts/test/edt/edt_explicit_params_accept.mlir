// RUN: %carts-compile %s --arts-config %arts_config --start-from initial-cleanup --pipeline initial-cleanup | %FileCheck %s

// CHECK-LABEL: func.func @explicit_edt_param_ok
// CHECK: arts.edt <task> <intranode>
// CHECK-SAME: params(%{{.*}} : index)
// CHECK: ^bb0(%{{.*}}: memref<?xmemref<?xi32>>, %[[P:.*]]: index):
// CHECK: arith.index_cast %[[P]] : index to i32

module attributes {arts.runtime_total_nodes = 1 : i64, arts.runtime_total_workers = 8 : i64} {
  func.func @explicit_edt_param_ok() {
    %route = arith.constant -1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %p = arith.constant 7 : index

    %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>, <coarse>] route(%route : i32) sizes[%c1] elementType(i32) elementSizes[%c1] : (memref<?xi64>, memref<?xmemref<?xi32>>)
    %acq_guid, %acq_ptr = arts.db_acquire[<inout>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<?xi32>>) partitioning(<coarse>), indices[], offsets[%c0], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?xi32>>)

    arts.edt <task> <intranode> route(%route) (%acq_ptr) : memref<?xmemref<?xi32>> params(%p : index) {
    ^bb0(%dep: memref<?xmemref<?xi32>>, %p_arg: index):
      %local0 = arith.constant 0 : index
      %payload = arts.db_ref %dep[%local0] : memref<?xmemref<?xi32>> -> memref<?xi32>
      %old = memref.load %payload[%local0] : memref<?xi32>
      %p_i32 = arith.index_cast %p_arg : index to i32
      %new = arith.addi %old, %p_i32 : i32
      memref.store %new, %payload[%local0] : memref<?xi32>
      arts.yield
    }

    arts.db_release(%acq_ptr) : memref<?xmemref<?xi32>>
    return
  }
}
