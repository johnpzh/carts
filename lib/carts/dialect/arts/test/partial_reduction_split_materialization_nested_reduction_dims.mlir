// RUN: %carts-compile %s --pass-pipeline='builtin.module(partial-reduction-split-materialization)' \
// RUN:   | %FileCheck %s --implicit-check-not=partialReductionSplitRequired

module {
  func.func @partial_reduction_split_materialization_nested_reduction_dims() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %route = arith.constant -1 : i32
    %result_guid, %result_ptr = arts.db_alloc[<inout>, <heap>, <write>, <block>]
      route(%route : i32) sizes[%c4] elementType(f32) elementSizes[%c1]
      {planLogicalWorkerSlice = [1], planOwnerDims = [0], planPhysicalBlockShape = [1]}
      : (memref<?xi64>, memref<?xmemref<?xf32>>)
    %input_guid, %input_ptr = arts.db_alloc[<in>, <heap>, <read>, <block>]
      route(%route : i32) sizes[%c4] elementType(f32) elementSizes[%c1, %c8, %c4]
      {planLogicalWorkerSlice = [1], planOwnerDims = [0], planPhysicalBlockShape = [1]}
      : (memref<?xi64>, memref<?xmemref<?x?x?xf32>>)
    %mean_guid, %mean_ptr = arts.db_alloc[<in>, <heap>, <read>, <block>]
      route(%route : i32) sizes[%c4] elementType(f32) elementSizes[%c1]
      {planLogicalWorkerSlice = [1], planOwnerDims = [0], planPhysicalBlockShape = [1]}
      : (memref<?xi64>, memref<?xmemref<?xf32>>)
    scf.for %owner = %c0 to %c4 step %c1 {
      %result_acq_guid, %result_acq = arts.db_acquire[<inout>]
        (%result_guid : memref<?xi64>, %result_ptr : memref<?xmemref<?xf32>>)
        partitioning(<block>), indices[], offsets[%owner], sizes[%c1]
        -> (memref<?xi64>, memref<?xmemref<?xf32>>)
      %input_acq_guid, %input_acq = arts.db_acquire[<in>]
        (%input_guid : memref<?xi64>, %input_ptr : memref<?xmemref<?x?x?xf32>>)
        partitioning(<block>), indices[], offsets[%owner], sizes[%c1]
        -> (memref<?xi64>, memref<?xmemref<?x?x?xf32>>)
      %mean_acq_guid, %mean_acq = arts.db_acquire[<in>]
        (%mean_guid : memref<?xi64>, %mean_ptr : memref<?xmemref<?xf32>>)
        partitioning(<block>), indices[], offsets[%owner], sizes[%c1]
        -> (memref<?xi64>, memref<?xmemref<?xf32>>)
      arts.edt <task> <intranode> route(%route) (%result_acq, %input_acq, %mean_acq)
          : memref<?xmemref<?xf32>>, memref<?xmemref<?x?x?xf32>>, memref<?xmemref<?xf32>>
          params(%owner : index)
          attributes {
            depPattern = #arts.dep_pattern<elementwise_pipeline>,
            partialReduction,
            partialReductionDepResultDimMaps = [[0], [0, -1, -1], [0]],
            partialReductionDims = [1, 2],
            partialReductionOwnerDims = [0],
            partialReductionSplitDims = [1, 2],
            partialReductionSplitFactor = 2 : i64,
            partialReductionSplitOwnerTaskCount = 4 : i64,
            partialReductionSplitRequired,
            partialReductionSplitTargetWorkerCount = 8 : i64,
            reductionStrategy = #arts.reduction_strategy<local_accumulate>
          } {
      ^bb0(%result_arg: memref<?xmemref<?xf32>>, %input_arg: memref<?xmemref<?x?x?xf32>>, %mean_arg: memref<?xmemref<?xf32>>, %owner_arg: index):
        %result = arts.db_ref %result_arg[%c0] : memref<?xmemref<?xf32>> -> memref<?xf32>
        %input = arts.db_ref %input_arg[%c0] : memref<?xmemref<?x?x?xf32>> -> memref<?x?x?xf32>
        %mean = arts.db_ref %mean_arg[%c0] : memref<?xmemref<?xf32>> -> memref<?xf32>
        %old = memref.load %result[%c0] : memref<?xf32>
        scf.for %j = %c0 to %c8 step %c1 {
          scf.for %k = %c0 to %c4 step %c1 {
            %value = memref.load %input[%c0, %j, %k] : memref<?x?x?xf32>
            %mean_value = memref.load %mean[%c0] : memref<?xf32>
            %adjusted = arith.addf %value, %mean_value : f32
            %next = arith.addf %old, %adjusted : f32
            memref.store %next, %result[%c0] : memref<?xf32>
          }
        }
      }
    }
    return
  }
}

// CHECK-LABEL: func.func @partial_reduction_split_materialization_nested_reduction_dims
// CHECK: %[[PARTIAL_GUID:[A-Za-z0-9_]+]], %[[PARTIAL_PTR:[A-Za-z0-9_]+]] = arts.db_alloc{{.*}}<block>{{.*}}sizes[{{.*}}, {{.*}}]{{.*}}depPattern = #arts.dep_pattern<reduction>
// CHECK: scf.for
// CHECK: arts.db_acquire[<out>] (%[[PARTIAL_GUID]] : {{.*}}, %[[PARTIAL_PTR]]
// CHECK: arts.edt <task>
// CHECK-SAME: partialReduction
// CHECK-SAME: partialReductionDims = [1, 2]
// CHECK-SAME: partialReductionSplitDims = [1, 2]
// CHECK: arith.divui
// CHECK: arith.minui
// CHECK: scf.for
// CHECK: scf.for
// CHECK: arith.addf
// CHECK: memref.store
// CHECK: arts.db_acquire[<in>] (%[[PARTIAL_GUID]] : {{.*}}, %[[PARTIAL_PTR]]
// CHECK: arts.edt <task>
// CHECK-SAME: reductionStrategy = #arts.reduction_strategy<local_accumulate>
