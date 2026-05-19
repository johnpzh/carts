// RUN: %carts-compile %s --pipeline post-db-refinement --start-from post-db-refinement --arts-config %inputs_dir/arts_multinode.cfg --distributed-db | %FileCheck %s --implicit-check-not=arts.local_only --implicit-check-not=arts.distributed_reject_reason

// Rank-1 block-planned DBs are valid distributed storage for vector outputs and
// intermediates. They are common in ATAX/BiCG and normalization kernels, so the
// distributed ownership gate must not reject them as unsupported vectors once
// SDE/CODIR has authored an explicit owner/block plan.

// CHECK-LABEL: func.func @rank1_block_alloc_can_be_distributed
// CHECK: arts.db_alloc
// CHECK-SAME: <block>
// CHECK-SAME: distributed
// CHECK-SAME: planLogicalWorkerSlice = [16]
// CHECK-SAME: planOwnerDims = [0]
// CHECK-SAME: planPhysicalBlockShape = [16]
// CHECK: arts.edt <task> <internode>

module {
  func.func @rank1_block_alloc_can_be_distributed() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %route = arith.constant 0 : i32
    %value = arith.constant 1.0 : f64

    %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>, <block>, <uniform>] route(%route : i32) sizes[%c8] elementType(f64) elementSizes[%c16] {distribution_kind = #arts.distribution_kind<block>, planLogicalWorkerSlice = [16], planOwnerDims = [0], planPhysicalBlockShape = [16]} : (memref<?xi64>, memref<?xmemref<?xf64>>)
    %acq_guid, %acq_ptr = arts.db_acquire[<inout>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<?xf64>>) partitioning(<block>), indices[], offsets[%c0], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?xf64>>)

    arts.edt <task> <internode> route(%route) (%acq_ptr) : memref<?xmemref<?xf64>> attributes {distribution_kind = #arts.distribution_kind<block>, planLogicalWorkerSlice = [16], planOwnerDims = [0], planPhysicalBlockShape = [16]} {
    ^bb0(%dep: memref<?xmemref<?xf64>>):
      %payload = arts.db_ref %dep[%c0] : memref<?xmemref<?xf64>> -> memref<?xf64>
      memref.store %value, %payload[%c0] : memref<?xf64>
      arts.db_release(%dep) : memref<?xmemref<?xf64>>
      arts.yield
    }

    arts.db_release(%acq_ptr) : memref<?xmemref<?xf64>>
    return
  }
}
