// RUN: %carts-compile %s --pass-pipeline='builtin.module(verify-arts-objects-only)' | %FileCheck %s

module attributes {arts.runtime_total_nodes = 4 : i64, arts.runtime_total_workers = 64 : i64} {
  func.func @internode_task_allows_small_readonly_coarse_db() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %route = arith.constant 0 : i32
    %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>, <coarse>] route(%route : i32) sizes[%c1] elementType(f64) elementSizes[%c128] : (memref<?xi64>, memref<?xmemref<?xf64>>)
    %acq_guid, %acq_ptr = arts.db_acquire[<in>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<?xf64>>) partitioning(<coarse>), indices[], offsets[%c0], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?xf64>>)
    // CHECK-LABEL: func.func @internode_task_allows_small_readonly_coarse_db
    // CHECK: arts.edt <task> <internode>
    arts.edt <task> <internode> route(%route) (%acq_ptr) : memref<?xmemref<?xf64>> attributes {distribution_kind = #arts.distribution_kind<block>, planLogicalWorkerSlice = [16], planOwnerDims = [0], planPhysicalBlockShape = [16]} {
    ^bb0(%arg0: memref<?xmemref<?xf64>>):
      arts.yield
    }
    func.return
  }
}
