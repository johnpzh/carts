// RUN: not %carts-compile %s --pass-pipeline='builtin.module(verify-arts-objects-only)' 2>&1 | %FileCheck %s

module attributes {arts.runtime_total_nodes = 4 : i64, arts.runtime_total_workers = 64 : i64} {
  func.func @internode_planned_task_rejects_coarse_db() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %route = arith.constant 0 : i32
    %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>, <coarse>] route(%route : i32) sizes[%c1] elementType(f64) elementSizes[%c128, %c128] : (memref<?xi64>, memref<?xmemref<?x?xf64>>)
    %acq_guid, %acq_ptr = arts.db_acquire[<in>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<?x?xf64>>) partitioning(<coarse>), indices[], offsets[%c0], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?x?xf64>>)
    // CHECK: internode ARTS task depends on a coarse single-block aggregate user DB
    arts.edt <task> <internode> route(%route) (%acq_ptr) : memref<?xmemref<?x?xf64>> attributes {planOwnerDims = [0, 1], planPhysicalBlockShape = [16, 16]} {
    ^bb0(%arg0: memref<?xmemref<?x?xf64>>):
      arts.yield
    }
    func.return
  }

  func.func @internode_task_without_block_shape_rejects_coarse_db() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %route = arith.constant 0 : i32
    %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>, <coarse>] route(%route : i32) sizes[%c1] elementType(f64) elementSizes[%c128] : (memref<?xi64>, memref<?xmemref<?xf64>>)
    %acq_guid, %acq_ptr = arts.db_acquire[<in>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<?xf64>>) partitioning(<coarse>), indices[], offsets[%c0], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?xf64>>)
    // CHECK: internode ARTS task depends on a coarse single-block aggregate user DB
    arts.edt <task> <internode> route(%route) (%acq_ptr) : memref<?xmemref<?xf64>> attributes {distribution_kind = #arts.distribution_kind<block>} {
    ^bb0(%arg0: memref<?xmemref<?xf64>>):
      arts.yield
    }
    func.return
  }
}
