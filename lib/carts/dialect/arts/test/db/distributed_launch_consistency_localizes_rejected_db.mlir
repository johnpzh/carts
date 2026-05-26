// RUN: %carts-compile %s --pass-pipeline='builtin.module(distributed-launch-consistency)' \
// RUN:   | %FileCheck %s

module attributes {arts.runtime_total_nodes = 4 : i64, arts.runtime_total_workers = 256 : i64} {
  func.func @localizes_rejected_block_db() {
    %route = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>, <block>] route(%route : i32) sizes[%c4] elementType(f64) elementSizes[%c1] {arts.distributed_reject_reason = "unsupported_ptr_users", local_only, planOwnerDims = [0], planPhysicalBlockShape = [1]} : (memref<?xi64>, memref<?xmemref<?xf64>>)
    %acq_guid, %acq_ptr = arts.db_acquire[<inout>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<?xf64>>) partitioning(<block>), indices[], offsets[%c0], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?xf64>>)
    arts.edt <task> <internode> route(%route) (%acq_ptr) : memref<?xmemref<?xf64>> {
    ^bb0(%arg0: memref<?xmemref<?xf64>>):
      arts.yield
    }
    return
  }

  func.func @keeps_small_readonly_coarse_db() {
    %route = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %guid, %ptr = arts.db_alloc[<in>, <heap>, <read>, <coarse>] route(%route : i32) sizes[%c1] elementType(f64) elementSizes[%c16] {arts.distributed_reject_reason = "single_block"} : (memref<?xi64>, memref<?xmemref<?xf64>>)
    %acq_guid, %acq_ptr = arts.db_acquire[<in>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<?xf64>>) partitioning(<coarse>), indices[], offsets[%c0], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?xf64>>)
    arts.edt <task> <internode> route(%route) (%acq_ptr) : memref<?xmemref<?xf64>> {
    ^bb0(%arg0: memref<?xmemref<?xf64>>):
      arts.yield
    }
    return
  }
}

// CHECK-LABEL: func.func @localizes_rejected_block_db
// CHECK: %[[LOCAL_ROUTE:.*]] = arith.constant -1 : i32
// CHECK: arts.edt <task> <intranode> route(%[[LOCAL_ROUTE]])

// CHECK-LABEL: func.func @keeps_small_readonly_coarse_db
// CHECK: arts.edt <task> <internode>
