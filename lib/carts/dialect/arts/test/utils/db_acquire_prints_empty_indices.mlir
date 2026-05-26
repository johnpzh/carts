// RUN: %carts-compile %s --pass-pipeline='builtin.module(cse)' | %FileCheck %s

// CHECK-LABEL: func.func @db_acquire_roundtrip_empty_indices
// CHECK: partitioning(<stencil>, offsets[%{{.*}}], sizes[%{{.*}}])
// CHECK-SAME: , indices[], offsets[%{{.*}}], sizes[%{{.*}}]
module {
  func.func @db_acquire_roundtrip_empty_indices() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %route = arith.constant -1 : i32
    %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>, <coarse>] route(%route : i32) sizes[%c1] elementType(f64) elementSizes[%c4] : (memref<?xi64>, memref<?xmemref<?xf64>>)
    %acq_guid, %acq_ptr = arts.db_acquire[<in>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<?xf64>>) partitioning(<stencil>, indices[], offsets[%c0], sizes[%c1]), indices[], offsets[%c0], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?xf64>>)
    return
  }
}
