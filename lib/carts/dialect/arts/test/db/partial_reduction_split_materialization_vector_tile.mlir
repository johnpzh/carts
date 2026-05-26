// RUN: %carts-compile %s --pass-pipeline='builtin.module(partial-reduction-split-materialization)' \
// RUN:   | %FileCheck %s --implicit-check-not=partialReductionSplitRequired

module {
  func.func @partial_reduction_split_materialization_vector_tile() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %route = arith.constant -1 : i32
    %zero = arith.constant 0.000000e+00 : f32
    %result_guid, %result_ptr = arts.db_alloc[<inout>, <heap>, <write>, <block>]
      route(%route : i32) sizes[%c4] elementType(f32) elementSizes[%c4]
      {planLogicalWorkerSlice = [4], planOwnerDims = [0], planPhysicalBlockShape = [4]}
      : (memref<?xi64>, memref<?xmemref<?xf32>>)
    %input_guid, %input_ptr = arts.db_alloc[<in>, <heap>, <read>, <block>]
      route(%route : i32) sizes[%c4] elementType(f32) elementSizes[%c4, %c8]
      {planLogicalWorkerSlice = [4], planOwnerDims = [0], planPhysicalBlockShape = [4]}
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
            partialReductionSplitFactor = 2 : i64,
            partialReductionSplitOwnerTaskCount = 4 : i64,
            partialReductionSplitRequired,
            partialReductionSplitTargetWorkerCount = 8 : i64,
            planLogicalWorkerSlice = [4],
            planOwnerDims = [0],
            planPhysicalBlockShape = [4],
            reductionStrategy = #arts.reduction_strategy<local_accumulate>
          } {
      ^bb0(%result_arg: memref<?xmemref<?xf32>>, %input_arg: memref<?xmemref<?x?xf32>>, %owner_arg: index):
        %result = arts.db_ref %result_arg[%c0] : memref<?xmemref<?xf32>> -> memref<?xf32>
        %input = arts.db_ref %input_arg[%c0] : memref<?xmemref<?x?xf32>> -> memref<?x?xf32>
        scf.for %i = %c0 to %c4 step %c1 {
          memref.store %zero, %result[%i] : memref<?xf32>
          scf.for %j = %c0 to %c8 step %c1 {
            %old = memref.load %result[%i] : memref<?xf32>
            %value = memref.load %input[%i, %j] : memref<?x?xf32>
            %next = arith.addf %old, %value : f32
            memref.store %next, %result[%i] : memref<?xf32>
          }
        }
      }
    }
    return
  }
}

// CHECK-LABEL: func.func @partial_reduction_split_materialization_vector_tile
// CHECK: %[[PARTIAL_GUID:[A-Za-z0-9_]+]], %[[PARTIAL_PTR:[A-Za-z0-9_]+]] = arts.db_alloc{{.*}}sizes[{{.*}}, {{.*}}]{{.*}}elementSizes[{{.*}}]{{.*}}depPattern = #arts.dep_pattern<reduction>{{.*}}planLogicalWorkerSlice = [4]
// CHECK: scf.for
// CHECK: arts.db_acquire[<out>] (%[[PARTIAL_GUID]] : {{.*}}, %[[PARTIAL_PTR]] : {{.*}}){{.*}}indices[{{.*}}, {{.*}}]
// CHECK: arts.edt <task>
// CHECK: scf.for
// CHECK: memref.store {{.*}} : memref<?xf32>
// CHECK: memref.load {{.*}} : memref<?x?xf32>
// CHECK: memref.load {{.*}} : memref<?xf32>
// CHECK: memref.store {{.*}} : memref<?xf32>
// CHECK: arts.edt <task>
// CHECK: scf.for
// CHECK: memref.load {{.*}} : memref<?xf32>
// CHECK: memref.load {{.*}} : memref<?xf32>
// CHECK: memref.store {{.*}} : memref<?xf32>
