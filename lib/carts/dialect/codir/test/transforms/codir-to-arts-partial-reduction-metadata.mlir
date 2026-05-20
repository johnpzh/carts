// RUN: %carts-compile %s --arts-config %inputs_dir/arts_multinode_8x64.cfg \
// RUN:   --pass-pipeline='builtin.module(reduction-planning,storage-planning,convert-codir-to-arts,verify-arts-objects-only)' \
// RUN:   | %FileCheck %s

module {
  func.func @partial_reduction_metadata_to_arts(%A: memref<128x128xf64>, %y: memref<128xf64>, %base: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    scf.for %i = %base to %c128 step %c16 {
      codir.codelet deps(%y, %A : memref<128xf64>, memref<128x128xf64>)
          params(%i : index)
          attributes {dep_modes = [#codir.access_mode<readwrite>, #codir.access_mode<read>],
                      dep_storage_views = [#codir.storage_view<compute_block>, #codir.storage_view<compute_block>],
                      distribution_kind = #codir.distribution_kind<blocked>,
                      iteration_topology = #codir.iteration_topology<owner_strip>,
                      logical_worker_slice = [16],
                      partial_reduction,
                      partial_reduction_dims = [1],
                      partial_reduction_owner_dims = [0],
                      pattern = #codir.pattern<elementwise_pipeline>,
                      tile_owner_dims = [0],
                      tile_shape = [16]} {
      ^bb0(%arg0: memref<128xf64>, %arg1: memref<128x128xf64>, %owner: index):
        %inner_c0 = arith.constant 0 : index
        %inner_c1 = arith.constant 1 : index
        %inner_c128 = arith.constant 128 : index
        %old = memref.load %arg0[%owner] : memref<128xf64>
        scf.for %j = %inner_c0 to %inner_c128 step %inner_c1 {
          %a = memref.load %arg1[%owner, %j] : memref<128x128xf64>
          %next = arith.addf %old, %a : f64
          memref.store %next, %arg0[%owner] : memref<128xf64>
        }
        codir.yield
      }
    }
    return
  }
}

// CHECK-LABEL: func.func @partial_reduction_metadata_to_arts
// CHECK: arts.edt <task>{{.*}}partialReduction, partialReductionDepResultDimMaps = {{\[\[}}0], [0, -1]]
// CHECK-SAME: partialReductionDims = [1]
// CHECK-SAME: partialReductionOwnerDims = [0]
// CHECK-SAME: reductionStrategy = #arts.reduction_strategy<local_accumulate>
