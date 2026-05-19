// RUN: %carts-compile %s --pass-pipeline='builtin.module(convert-codir-to-arts)' \
// RUN:   --arts-config %inputs_dir/arts_multinode_8x64.cfg | %FileCheck %s

// Even with an owner/block plan and owner-local indexing, host_whole is not a
// compute-block storage contract.

module {
  func.func @host_whole_storage_view_stays_coarse(%A: memref<128xf32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    scf.for %i = %c0 to %n step %c16 {
      codir.codelet deps(%A : memref<128xf32>) params(%i : index)
          attributes {dep_modes = [#codir.access_mode<readwrite>],
                      dep_storage_views = [#codir.storage_view<host_whole>],
                      distribution_kind = #codir.distribution_kind<owner_compute>,
                      iteration_topology = #codir.iteration_topology<owner_strip>,
                      logical_worker_slice = [16],
                      tile_owner_dims = [0],
                      tile_shape = [16]} {
      ^bb0(%arg0: memref<128xf32>, %base: index):
        %value = memref.load %arg0[%base] : memref<128xf32>
        memref.store %value, %arg0[%base] : memref<128xf32>
        codir.yield
      }
    }
    return
  }
}

// CHECK-LABEL: func.func @host_whole_storage_view_stays_coarse
// CHECK: arts.db_alloc
// CHECK-SAME: <coarse>
// CHECK: arts.edt <task> <intranode>
// CHECK-NOT: arts.edt <task> <internode>
