// RUN: %carts-compile %s --pass-pipeline='builtin.module(verify-codir,convert-codir-to-arts)' \
// RUN:   --arts-config %inputs_dir/arts_multinode_4x16.cfg | %FileCheck %s --implicit-check-not="arts.edt <task> <intranode>"

module attributes {arts.runtime_total_nodes = 4 : i64, arts.runtime_total_workers = 64 : i64} {
  func.func @small_readonly_host_whole_dep_does_not_force_local() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %A = memref.alloc() : memref<128xf32>
    %scale = memref.alloc() : memref<128xf32>
    scf.for %i = %c0 to %c128 step %c16 {
      codir.codelet deps(%A, %scale : memref<128xf32>, memref<128xf32>) params(%i : index)
          attributes {dep_modes = [#codir.access_mode<readwrite>, #codir.access_mode<read>],
                      dep_storage_views = [#codir.storage_view<compute_block>, #codir.storage_view<host_whole>],
                      distribution_kind = #codir.distribution_kind<blocked>,
                      iteration_topology = #codir.iteration_topology<owner_strip>,
                      logical_worker_slice = [16],
                      pattern = #codir.pattern<uniform>,
                      tile_owner_dims = [0],
                      tile_shape = [16]} {
      ^bb0(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %base: index):
        %inner_c1 = arith.constant 1 : index
        %inner_c16 = arith.constant 16 : index
        %inner_c128 = arith.constant 128 : index
        %end_raw = arith.addi %base, %inner_c16 : index
        %end = arith.minui %end_raw, %inner_c128 : index
        scf.for %j = %base to %end step %inner_c1 {
          %value = memref.load %arg0[%j] : memref<128xf32>
          %factor = memref.load %arg1[%j] : memref<128xf32>
          %next = arith.mulf %value, %factor : f32
          memref.store %next, %arg0[%j] : memref<128xf32>
        }
        codir.yield
      }
    }
    memref.dealloc %scale : memref<128xf32>
    memref.dealloc %A : memref<128xf32>
    return
  }
}

// CHECK-LABEL: func.func @small_readonly_host_whole_dep_does_not_force_local
// CHECK: arts.db_alloc{{.*}}<block>
// CHECK: arts.db_alloc{{.*}}<coarse>
// CHECK: arts.edt <task> <internode>
