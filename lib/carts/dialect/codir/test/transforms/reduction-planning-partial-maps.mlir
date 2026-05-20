// RUN: %carts-compile %s --pass-pipeline='builtin.module(reduction-planning,verify-codir)' \
// RUN:   | %FileCheck %s

module {
  func.func @partial_reduction_maps(%A: memref<128x128xf64>, %x: memref<128xf64>, %y: memref<128xf64>, %base: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    scf.for %i = %base to %c128 step %c16 {
      codir.codelet deps(%y, %A, %x : memref<128xf64>, memref<128x128xf64>, memref<128xf64>)
          params(%i : index)
          attributes {dep_modes = [#codir.access_mode<readwrite>, #codir.access_mode<read>, #codir.access_mode<read>],
                      dep_storage_views = [#codir.storage_view<compute_block>, #codir.storage_view<compute_block>, #codir.storage_view<host_whole>],
                      partial_reduction,
                      partial_reduction_dims = [1],
                      partial_reduction_owner_dims = [0],
                      tile_owner_dims = [0],
                      tile_shape = [16]} {
      ^bb0(%arg0: memref<128xf64>, %arg1: memref<128x128xf64>, %arg2: memref<128xf64>, %owner: index):
        %inner_c0 = arith.constant 0 : index
        %inner_c1 = arith.constant 1 : index
        %inner_c128 = arith.constant 128 : index
        %old = memref.load %arg0[%owner] : memref<128xf64>
        scf.for %j = %inner_c0 to %inner_c128 step %inner_c1 {
          %a = memref.load %arg1[%owner, %j] : memref<128x128xf64>
          %v = memref.load %arg2[%j] : memref<128xf64>
          %prod = arith.mulf %a, %v : f64
          %next = arith.addf %old, %prod : f64
          memref.store %next, %arg0[%owner] : memref<128xf64>
        }
        codir.yield
      }
    }
    return
  }
}

// CHECK-LABEL: func.func @partial_reduction_maps
// CHECK: codir.codelet
// CHECK-SAME: partial_reduction
// CHECK-SAME: partial_reduction_dep_result_dim_maps = {{\[\[}}0], [0, -1], [-1]]
// CHECK-SAME: reduction_strategy = #codir.reduction_strategy<local_accumulate>
