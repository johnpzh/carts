// RUN: %carts-compile %s --pass-pipeline='builtin.module(partial-reduction-split-materialization)' \
// RUN:   | %FileCheck %s --implicit-check-not=partialReductionSplitRequired --implicit-check-not=local_only

module attributes {arts.runtime_total_nodes = 4 : i64, arts.runtime_total_workers = 12 : i64} {
  func.func @partial_reduction_split_materialization_multinode_routes() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %route = arith.constant -1 : i32
    %result_guid, %result_ptr = arts.db_alloc[<inout>, <heap>, <write>, <block>]
      route(%route : i32) sizes[%c4] elementType(f32) elementSizes[%c1]
      {planLogicalWorkerSlice = [1], planOwnerDims = [0], planPhysicalBlockShape = [1]}
      : (memref<?xi64>, memref<?xmemref<?xf32>>)
    %input_guid, %input_ptr = arts.db_alloc[<in>, <heap>, <read>, <block>]
      route(%route : i32) sizes[%c4] elementType(f32) elementSizes[%c1, %c8]
      {planLogicalWorkerSlice = [1], planOwnerDims = [0], planPhysicalBlockShape = [1]}
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
            distribution_kind = #arts.distribution_kind<block>,
            distribution_pattern = #arts.distribution_pattern<uniform>,
            distribution_version = 1 : i32,
            partialReduction,
            partialReductionDepResultDimMaps = [[0], [0, -1]],
            partialReductionDims = [1],
            partialReductionOwnerDims = [0],
            partialReductionSplitDims = [1],
            partialReductionSplitFactor = 3 : i64,
            partialReductionSplitOwnerTaskCount = 4 : i64,
            partialReductionSplitRequired,
            partialReductionSplitTargetWorkerCount = 12 : i64,
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

// CHECK-LABEL: func.func @partial_reduction_split_materialization_multinode_routes
// CHECK: %[[PARTIAL_GUID:[A-Za-z0-9_]+]], %[[PARTIAL_PTR:[A-Za-z0-9_]+]] = arts.db_alloc{{.*}}sizes[{{.*}}, {{.*}}]{{.*}}{depPattern = #arts.dep_pattern<reduction>, distributed
// CHECK-SAME: distribution_kind = #arts.distribution_kind<block_cyclic>
// CHECK: %[[TREE_GUID:[A-Za-z0-9_]+]], %[[TREE_PTR:[A-Za-z0-9_]+]] = arts.db_alloc{{.*}}sizes[{{.*}}, {{.*}}]{{.*}}{depPattern = #arts.dep_pattern<reduction>, distributed
// CHECK-SAME: distribution_kind = #arts.distribution_kind<block_cyclic>
// CHECK: scf.for %[[OWNER:[A-Za-z0-9_]+]]
// CHECK: scf.for %[[TILE:[A-Za-z0-9_]+]]
// CHECK: arts.db_acquire[<out>] (%[[PARTIAL_GUID]] : {{.*}}, %[[PARTIAL_PTR]]
// CHECK: %[[REL:.*]] = arith.subi %[[OWNER]],
// CHECK: %[[ORD:.*]] = arith.divui %[[REL]],
// CHECK: %[[BASE:.*]] = arith.muli %[[ORD]],
// CHECK: %[[LINEAR:.*]] = arith.addi %[[BASE]], %[[TILE]]
// CHECK: %[[LINEAR_I32:.*]] = arith.index_cast %[[LINEAR]] : index to i32
// CHECK: %[[NODES:.*]] = arts.runtime_query <total_nodes> -> i32
// CHECK: %[[ROUTE:.*]] = arith.remui %[[LINEAR_I32]], %[[NODES]] : i32
// CHECK: arts.edt <task> <internode> route(%[[ROUTE]])
// CHECK-SAME: distribution_kind = #arts.distribution_kind<block_cyclic>
// CHECK-SAME: partialReduction
// CHECK: arts.db_acquire[<out>] (%[[TREE_GUID]] : {{.*}}, %[[TREE_PTR]]
// CHECK: arts.db_acquire[<in>] (%[[PARTIAL_GUID]] : {{.*}}, %[[PARTIAL_PTR]]
// CHECK: arts.edt <task> <internode> route(
// CHECK-SAME: distribution_kind = #arts.distribution_kind<block_cyclic>
// CHECK: arts.db_acquire[<in>] (%[[TREE_GUID]] : {{.*}}, %[[TREE_PTR]]
// CHECK: arts.edt <task> <internode> route(
