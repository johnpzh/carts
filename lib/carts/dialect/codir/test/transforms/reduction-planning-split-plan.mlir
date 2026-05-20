// RUN: %carts-compile %s --pass-pipeline='builtin.module(reduction-planning,verify-codir)' \
// RUN:   | %FileCheck %s --check-prefix=CODIR
// RUN: %carts-compile %s --pass-pipeline='builtin.module(reduction-planning,storage-planning,convert-codir-to-arts,verify-arts-objects-only)' \
// RUN:   | %FileCheck %s --check-prefix=ARTS
// RUN: %carts-compile %s --pass-pipeline='builtin.module(reduction-planning,storage-planning,convert-codir-to-arts,partial-reduction-split-materialization,verify-arts-objects-only)' \
// RUN:   | %FileCheck %s --check-prefix=MATERIALIZED --implicit-check-not=partialReductionSplitRequired

module attributes {arts.runtime_total_workers = 4096 : i64} {
  func.func @partial_reduction_split_plan(%A: memref<1920x16xf32>, %y: memref<1920xf32>, %base: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c1920 = arith.constant 1920 : index
    scf.for %i = %base to %c1920 step %c1 {
      codir.codelet deps(%y, %A : memref<1920xf32>, memref<1920x16xf32>)
          params(%i : index)
          attributes {dep_modes = [#codir.access_mode<readwrite>, #codir.access_mode<read>],
                      dep_storage_views = [#codir.storage_view<compute_block>, #codir.storage_view<compute_block>],
                      distribution_kind = #codir.distribution_kind<blocked>,
                      iteration_topology = #codir.iteration_topology<owner_strip>,
                      logical_worker_slice = [1],
                      partial_reduction,
                      partial_reduction_dims = [1],
                      partial_reduction_owner_dims = [0],
                      pattern = #codir.pattern<elementwise_pipeline>,
                      tile_owner_dims = [0],
                      tile_shape = [1]} {
      ^bb0(%arg0: memref<1920xf32>, %arg1: memref<1920x16xf32>, %owner: index):
        %inner_c0 = arith.constant 0 : index
        %inner_c1 = arith.constant 1 : index
        %inner_c16 = arith.constant 16 : index
        %old = memref.load %arg0[%owner] : memref<1920xf32>
        scf.for %j = %inner_c0 to %inner_c16 step %inner_c1 {
          %a = memref.load %arg1[%owner, %j] : memref<1920x16xf32>
          %next = arith.addf %old, %a : f32
          memref.store %next, %arg0[%owner] : memref<1920xf32>
        }
        codir.yield
      }
    }
    return
  }
}

// CODIR-LABEL: func.func @partial_reduction_split_plan
// CODIR: codir.codelet
// CODIR-SAME: partial_reduction
// CODIR-SAME: partial_reduction_dep_result_dim_maps = {{\[\[}}0], [0, -1]]
// CODIR-SAME: partial_reduction_dims = [1]
// CODIR-SAME: partial_reduction_owner_dims = [0]
// CODIR-SAME: partial_reduction_split_dims = [1]
// CODIR-SAME: partial_reduction_split_factor = 3 : i64
// CODIR-SAME: partial_reduction_split_owner_task_count = 1920 : i64
// CODIR-SAME: partial_reduction_split_required
// CODIR-SAME: partial_reduction_split_target_worker_count = 4096 : i64

// ARTS-LABEL: func.func @partial_reduction_split_plan
// ARTS: arts.edt <task>{{.*}}partialReduction, partialReductionDepResultDimMaps = {{\[\[}}0], [0, -1]]
// ARTS-SAME: partialReductionDims = [1]
// ARTS-SAME: partialReductionOwnerDims = [0]
// ARTS-SAME: partialReductionSplitDims = [1]
// ARTS-SAME: partialReductionSplitFactor = 3 : i64
// ARTS-SAME: partialReductionSplitOwnerTaskCount = 1920 : i64
// ARTS-SAME: partialReductionSplitRequired
// ARTS-SAME: partialReductionSplitTargetWorkerCount = 4096 : i64

// MATERIALIZED-LABEL: func.func @partial_reduction_split_plan
// MATERIALIZED: %[[PARTIAL_GUID:[A-Za-z0-9_]+]], %[[PARTIAL_PTR:[A-Za-z0-9_]+]] = arts.db_alloc{{.*}}<block>{{.*}}sizes[{{.*}}, {{.*}}]{{.*}}depPattern = #arts.dep_pattern<reduction>
// MATERIALIZED: %[[TREE_GUID:[A-Za-z0-9_]+]], %[[TREE_PTR:[A-Za-z0-9_]+]] = arts.db_alloc{{.*}}<block>{{.*}}sizes[{{.*}}, {{.*}}]{{.*}}depPattern = #arts.dep_pattern<reduction>
// MATERIALIZED: scf.for
// MATERIALIZED: arts.db_acquire[<out>] (%[[PARTIAL_GUID]] : {{.*}}, %[[PARTIAL_PTR]]
// MATERIALIZED: arts.edt <task>
// MATERIALIZED-SAME: partialReduction
// MATERIALIZED-SAME: partialReductionSplitFactor = 3 : i64
// MATERIALIZED: arith.divui
// MATERIALIZED: arith.minui
// MATERIALIZED: memref.store
// MATERIALIZED: arts.db_acquire[<out>] (%[[TREE_GUID]] : {{.*}}, %[[TREE_PTR]]
// MATERIALIZED: arts.db_acquire[<in>] (%[[PARTIAL_GUID]] : {{.*}}, %[[PARTIAL_PTR]]
// MATERIALIZED: arts.edt <task>
// MATERIALIZED-SAME: reductionStrategy = #arts.reduction_strategy<local_accumulate>
// MATERIALIZED: arts.db_acquire[<in>] (%[[TREE_GUID]] : {{.*}}, %[[TREE_PTR]]
// MATERIALIZED: arts.edt <task>
// MATERIALIZED-SAME: reductionStrategy = #arts.reduction_strategy<local_accumulate>
