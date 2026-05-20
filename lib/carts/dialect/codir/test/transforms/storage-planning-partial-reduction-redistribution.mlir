// RUN: %carts-compile %s --pass-pipeline='builtin.module(reduction-planning,storage-planning,verify-codir)' \
// RUN:   | %FileCheck %s

module {
  func.func @partial_reduction_defers_huge_source_redistribution(%A: memref<1024x32768xf32>, %y: memref<32768xf32>, %base: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c1024 = arith.constant 1024 : index
    %c32768 = arith.constant 32768 : index
    scf.for %i = %base to %c32768 step %c16 {
      codir.codelet deps(%y, %A : memref<32768xf32>, memref<1024x32768xf32>)
          params(%i : index)
          attributes {dep_modes = [#codir.access_mode<readwrite>, #codir.access_mode<read>],
                      dep_storage_views = [#codir.storage_view<compute_block>, #codir.storage_view<compute_block>],
                      partial_reduction,
                      partial_reduction_dims = [1],
                      partial_reduction_owner_dims = [0],
                      tile_owner_dims = [0],
                      tile_shape = [16]} {
      ^bb0(%arg0: memref<32768xf32>, %arg1: memref<1024x32768xf32>, %owner: index):
        %inner_c0 = arith.constant 0 : index
        %inner_c1 = arith.constant 1 : index
        %inner_c1024 = arith.constant 1024 : index
        %old = memref.load %arg0[%owner] : memref<32768xf32>
        scf.for %j = %inner_c0 to %inner_c1024 step %inner_c1 {
          %a = memref.load %arg1[%j, %owner] : memref<1024x32768xf32>
          %next = arith.addf %old, %a : f32
          memref.store %next, %arg0[%owner] : memref<32768xf32>
        }
        codir.yield
      }
    }
    return
  }
}

// CHECK-LABEL: func.func @partial_reduction_defers_huge_source_redistribution
// CHECK: codir.codelet
// CHECK-SAME: dep_owner_dims = [{{\[}}0], [1]]
// CHECK-SAME: dep_storage_views = [#codir.storage_view<phase_redistributed>, #codir.storage_view<host_whole>]
// CHECK-SAME: partial_reduction_dep_result_dim_maps = {{\[\[}}0], [-1, 0]]
