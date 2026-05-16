// RUN: not %carts-compile %s --arts-config %arts_config --start-from initial-cleanup --pipeline initial-cleanup 2>&1 | %FileCheck %s

// CHECK: partition_offsets_segments segment sum (3) must match operand count (2)

module attributes {arts.runtime_total_nodes = 1 : i64, arts.runtime_total_workers = 4 : i64} {
  func.func @invalid_partition_segments() -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %route = arith.constant -1 : i32

    %guid, %ptr = arts.db_alloc[<in>, <heap>, <read>, <stencil>, <stencil>] route(%route : i32) sizes[%c4] elementType(f64) elementSizes[%c4] : (memref<?xi64>, memref<?xmemref<?xf64>>)
    %acq_guid, %acq_ptr = arts.db_acquire[<in>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<?xf64>>) partitioning(<stencil>, indices[], offsets[%c0, %c2], sizes[%c1, %c1]), indices[] {partition_offsets_segments = array<i32: 1, 2>, partition_sizes_segments = array<i32: 1, 1>} -> (memref<?xi64>, memref<?xmemref<?xf64>>)
    arts.db_release(%acq_ptr) : memref<?xmemref<?xf64>>
    arts.db_free(%guid) : memref<?xi64>
    arts.db_free(%ptr) : memref<?xmemref<?xf64>>
    %ret = arith.constant 0 : i32
    return %ret : i32
  }
}
