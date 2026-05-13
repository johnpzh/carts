// RUN: %carts-compile %s --arts-config %arts_config --start-from post-db-refinement --pipeline post-db-refinement | %FileCheck %s

// Rank-0 i1 tensor carriers are compiler control tokens.  When a task only
// writes constant true through such a dependency and never reads it, the slot
// should not become a real ARTS dependency that serializes otherwise
// independent workers.

// CHECK-LABEL: func.func @drops_true_only_control_dependency
// CHECK-NOT: memref.store %true
// CHECK: arts.edt <task> <intranode> route({{.*}}) (%{{[^)]*}}) : memref<?xmemref<?xf64>>
// CHECK: memref.load

module {
  func.func @drops_true_only_control_dependency() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c-1_i32 = arith.constant -1 : i32
    %true = arith.constant true
    %ctrl_guid, %ctrl_ptr = arts.db_alloc[<inout>, <heap>, <write>, <coarse>, <uniform>] route(%c-1_i32 : i32) sizes[%c1] elementType(i1) elementSizes[%c1] : (memref<?xi64>, memref<?xmemref<?xi1>>)
    %data_guid, %data_ptr = arts.db_alloc[<inout>, <heap>, <write>, <coarse>, <uniform>] route(%c-1_i32 : i32) sizes[%c1] elementType(f64) elementSizes[%c1] : (memref<?xi64>, memref<?xmemref<?xf64>>)
    %ctrl_acq_guid, %ctrl_acq_ptr = arts.db_acquire[<out>] (%ctrl_guid : memref<?xi64>, %ctrl_ptr : memref<?xmemref<?xi1>>) partitioning(<coarse>), indices[], offsets[%c0], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?xi1>>)
    %data_acq_guid, %data_acq_ptr = arts.db_acquire[<inout>] (%data_guid : memref<?xi64>, %data_ptr : memref<?xmemref<?xf64>>) partitioning(<coarse>), indices[], offsets[%c0], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?xf64>>)
    arts.edt <task> <intranode> route(%c-1_i32) (%ctrl_acq_ptr, %data_acq_ptr) : memref<?xmemref<?xi1>>, memref<?xmemref<?xf64>> {
    ^bb0(%ctrl: memref<?xmemref<?xi1>>, %data: memref<?xmemref<?xf64>>):
      %ctrl_view = arts.db_ref %ctrl[%c0] : memref<?xmemref<?xi1>> -> memref<?xi1>
      memref.store %true, %ctrl_view[%c0] : memref<?xi1>
      %data_view = arts.db_ref %data[%c0] : memref<?xmemref<?xf64>> -> memref<?xf64>
      %value = memref.load %data_view[%c0] : memref<?xf64>
      memref.store %value, %data_view[%c0] : memref<?xf64>
      arts.db_release(%ctrl) : memref<?xmemref<?xi1>>
      arts.db_release(%data) : memref<?xmemref<?xf64>>
    }
    arts.db_release(%ctrl_acq_ptr) : memref<?xmemref<?xi1>>
    arts.db_release(%data_acq_ptr) : memref<?xmemref<?xf64>>
    arts.db_free(%ctrl_guid) : memref<?xi64>
    arts.db_free(%ctrl_ptr) : memref<?xmemref<?xi1>>
    arts.db_free(%data_guid) : memref<?xi64>
    arts.db_free(%data_ptr) : memref<?xmemref<?xf64>>
    return
  }
}
