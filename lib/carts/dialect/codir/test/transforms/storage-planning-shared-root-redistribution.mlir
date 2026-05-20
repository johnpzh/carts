// RUN: %carts-compile %s --pass-pipeline='builtin.module(storage-planning,verify-codir)' \
// RUN:   | %FileCheck %s

module {
  func.func @shared_root_block_writer_before_whole_reader() {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %root = sde.mu_alloc : memref<128xf32>
    scf.for %i = %c0 to %c128 step %c16 {
      codir.codelet deps(%root : memref<128xf32>)
          params(%c16, %i : index, index)
          attributes {dep_modes = [#codir.access_mode<readwrite>],
                      dep_storage_views = [#codir.storage_view<compute_block>],
                      tile_owner_dims = [0],
                      tile_shape = [16]} {
      ^bb0(%arg0: memref<128xf32>, %step: index, %base: index):
        %inner_c1 = arith.constant 1 : index
        %value = arith.constant 1.000000e+00 : f32
        %ub = arith.addi %base, %step : index
        scf.for %j = %base to %ub step %inner_c1 {
          memref.store %value, %arg0[%j] : memref<128xf32>
        }
        codir.yield
      }
    }
    codir.codelet deps(%root : memref<128xf32>)
        params(%c0 : index)
        attributes {dep_modes = [#codir.access_mode<read>],
                    dep_storage_views = [#codir.storage_view<host_whole>]} {
    ^bb0(%arg0: memref<128xf32>, %idx: index):
      %unused = memref.load %arg0[%idx] : memref<128xf32>
      codir.yield
    }
    return
  }
}

// CHECK-LABEL: func.func @shared_root_block_writer_before_whole_reader
// CHECK: codir.codelet
// CHECK-SAME: dep_storage_views = [#codir.storage_view<phase_redistributed>]
// CHECK: codir.codelet
// CHECK-SAME: dep_storage_views = [#codir.storage_view<host_whole>]
