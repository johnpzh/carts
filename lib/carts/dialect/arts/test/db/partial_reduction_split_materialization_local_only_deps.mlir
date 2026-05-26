// RUN: %carts-compile %s --pass-pipeline='builtin.module(partial-reduction-split-materialization)' \
// RUN:   | %FileCheck %s --implicit-check-not=partialReductionSplitRequired --implicit-check-not=', distributed'

module attributes {arts.runtime_total_nodes = 4 : i64, arts.runtime_total_workers = 16 : i64} {
  func.func @local_only_dep_skips_unneeded_multinode_split() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %route = arith.constant -1 : i32
    %result_guid, %result_ptr = arts.db_alloc[<inout>, <heap>, <write>, <block>]
      route(%route : i32) sizes[%c4] elementType(f32) elementSizes[%c1]
      {arts.distributed_reject_reason = "unsupported_ptr_users", arts.local_only,
       planLogicalWorkerSlice = [1], planOwnerDims = [0], planPhysicalBlockShape = [1]}
      : (memref<?xi64>, memref<?xmemref<?xf32>>)
    %input_guid, %input_ptr = arts.db_alloc[<in>, <heap>, <read>, <block>]
      route(%route : i32) sizes[%c4] elementType(f32) elementSizes[%c1, %c8]
      {arts.distributed_reject_reason = "unsupported_ptr_users", arts.local_only,
       planLogicalWorkerSlice = [1], planOwnerDims = [0], planPhysicalBlockShape = [1]}
      : (memref<?xi64>, memref<?xmemref<?x?xf32>>)
    scf.for %owner = %c0 to %c4 step %c1 {
      %result_acq_guid, %result_acq = arts.db_acquire[<inout>]
        (%result_guid : memref<?xi64>, %result_ptr : memref<?xmemref<?xf32>>)
        partitioning(<block>), indices[], offsets[%owner], sizes[%c1]
        -> (memref<?xi64>, memref<?xmemref<?xf32>>)
      %input_acq_guid, %input_acq = arts.db_acquire[<in>]
        (%input_guid : memref<?xi64>, %input_ptr : memref<?xmemref<?x?xf32>>)
        partitioning(<block>), indices[], offsets[%owner], sizes[%c1]
        -> (memref<?xi64>, memref<?xmemref<?x?xf32>>)
      arts.edt <task> <intranode> route(%route) (%result_acq, %input_acq)
          : memref<?xmemref<?xf32>>, memref<?xmemref<?x?xf32>>
          params(%owner : index)
          attributes {
            depPattern = #arts.dep_pattern<elementwise_pipeline>,
            partialReduction,
            partialReductionDepResultDimMaps = [[0], [0, -1]],
            partialReductionDims = [1],
            partialReductionOwnerDims = [0],
            partialReductionSplitDims = [1],
            partialReductionSplitFactor = 4 : i64,
            partialReductionSplitOwnerTaskCount = 4 : i64,
            partialReductionSplitRequired,
            partialReductionSplitTargetWorkerCount = 16 : i64,
            reductionStrategy = #arts.reduction_strategy<local_accumulate>
          } {
      ^bb0(%result_arg: memref<?xmemref<?xf32>>, %input_arg: memref<?xmemref<?x?xf32>>, %owner_arg: index):
        %result = arts.db_ref %result_arg[%c0] : memref<?xmemref<?xf32>> -> memref<?xf32>
        %input = arts.db_ref %input_arg[%c0] : memref<?xmemref<?x?xf32>> -> memref<?x?xf32>
        %old = memref.load %result[%c0] : memref<?xf32>
        scf.for %j = %c0 to %c8 step %c1 {
          %value = memref.load %input[%c0, %j] : memref<?x?xf32>
          %next = arith.addf %old, %value : f32
          memref.store %next, %result[%c0] : memref<?xf32>
        }
      }
    }
    return
  }

  func.func @local_only_dep_caps_split_to_per_node_workers() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %route = arith.constant -1 : i32
    %result_guid, %result_ptr = arts.db_alloc[<inout>, <heap>, <write>, <block>]
      route(%route : i32) sizes[%c2] elementType(f32) elementSizes[%c1]
      {arts.distributed_reject_reason = "unsupported_ptr_users", arts.local_only,
       planLogicalWorkerSlice = [1], planOwnerDims = [0], planPhysicalBlockShape = [1]}
      : (memref<?xi64>, memref<?xmemref<?xf32>>)
    %input_guid, %input_ptr = arts.db_alloc[<in>, <heap>, <read>, <block>]
      route(%route : i32) sizes[%c2] elementType(f32) elementSizes[%c1, %c8]
      {arts.distributed_reject_reason = "unsupported_ptr_users", arts.local_only,
       planLogicalWorkerSlice = [1], planOwnerDims = [0], planPhysicalBlockShape = [1]}
      : (memref<?xi64>, memref<?xmemref<?x?xf32>>)
    scf.for %owner = %c0 to %c2 step %c1 {
      %result_acq_guid, %result_acq = arts.db_acquire[<inout>]
        (%result_guid : memref<?xi64>, %result_ptr : memref<?xmemref<?xf32>>)
        partitioning(<block>), indices[], offsets[%owner], sizes[%c1]
        -> (memref<?xi64>, memref<?xmemref<?xf32>>)
      %input_acq_guid, %input_acq = arts.db_acquire[<in>]
        (%input_guid : memref<?xi64>, %input_ptr : memref<?xmemref<?x?xf32>>)
        partitioning(<block>), indices[], offsets[%owner], sizes[%c1]
        -> (memref<?xi64>, memref<?xmemref<?x?xf32>>)
      arts.edt <task> <intranode> route(%route) (%result_acq, %input_acq)
          : memref<?xmemref<?xf32>>, memref<?xmemref<?x?xf32>>
          params(%owner : index)
          attributes {
            depPattern = #arts.dep_pattern<elementwise_pipeline>,
            partialReduction,
            partialReductionDepResultDimMaps = [[0], [0, -1]],
            partialReductionDims = [1],
            partialReductionOwnerDims = [0],
            partialReductionSplitDims = [1],
            partialReductionSplitFactor = 8 : i64,
            partialReductionSplitOwnerTaskCount = 2 : i64,
            partialReductionSplitRequired,
            partialReductionSplitTargetWorkerCount = 16 : i64,
            reductionStrategy = #arts.reduction_strategy<local_accumulate>
          } {
      ^bb0(%result_arg: memref<?xmemref<?xf32>>, %input_arg: memref<?xmemref<?x?xf32>>, %owner_arg: index):
        %result = arts.db_ref %result_arg[%c0] : memref<?xmemref<?xf32>> -> memref<?xf32>
        %input = arts.db_ref %input_arg[%c0] : memref<?xmemref<?x?xf32>> -> memref<?x?xf32>
        %old = memref.load %result[%c0] : memref<?xf32>
        scf.for %j = %c0 to %c8 step %c1 {
          %value = memref.load %input[%c0, %j] : memref<?x?xf32>
          %next = arith.addf %old, %value : f32
          memref.store %next, %result[%c0] : memref<?xf32>
        }
      }
    }
    return
  }
}

// CHECK-LABEL: func.func @local_only_dep_skips_unneeded_multinode_split
// CHECK-NOT: partialReductionSplit
// CHECK: arts.edt <task> <intranode>
// CHECK-SAME: partialReduction
// CHECK: return

// CHECK-LABEL: func.func @local_only_dep_caps_split_to_per_node_workers
// CHECK: arts.db_alloc{{.*}}sizes[{{.*}}, {{.*}}]{{.*}}{depPattern = #arts.dep_pattern<reduction>
// CHECK: scf.for
// CHECK: scf.for
// CHECK: arts.edt <task> <intranode> route(
// CHECK-SAME: partialReductionSplitFactor = 2 : i64
// CHECK-SAME: partialReductionSplitTargetWorkerCount = 4 : i64
