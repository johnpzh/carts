// RUN: %carts-compile %s --arts-config %inputs_dir/arts_multinode_8x64.cfg \
// RUN:   --pass-pipeline='builtin.module(convert-codir-to-arts,verify-arts-objects-only)' \
// RUN:   | %FileCheck %s

module attributes {arts.runtime_total_nodes = 4 : i64, arts.runtime_total_workers = 256 : i64} {
  func.func @compute_block_intermediate_survives_across_codelets() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %root = sde.mu_alloc : memref<64x32xf32>
    scf.for %i = %c0 to %c64 step %c8 {
      codir.codelet deps(%root : memref<64x32xf32>)
          params(%c8, %i : index, index)
          attributes {dep_modes = [#codir.access_mode<readwrite>],
                      dep_storage_views = [#codir.storage_view<compute_block>],
                      distribution_kind = #codir.distribution_kind<blocked>,
                      iteration_topology = #codir.iteration_topology<owner_strip>,
                      logical_worker_slice = [8, 32],
                      pattern = #codir.pattern<uniform>,
                      tile_owner_dims = [0],
                      tile_shape = [8, 32]} {
      ^bb0(%arg0: memref<64x32xf32>, %step: index, %base: index):
        %inner_c0 = arith.constant 0 : index
        %inner_c1 = arith.constant 1 : index
        %inner_c32 = arith.constant 32 : index
        %inner_cst = arith.constant 1.000000e+00 : f32
        %end = arith.addi %base, %step : index
        scf.for %row = %base to %end step %inner_c1 {
          scf.for %col = %inner_c0 to %inner_c32 step %inner_c1 {
            memref.store %inner_cst, %arg0[%row, %col] : memref<64x32xf32>
          }
        }
        codir.yield
      }
    }
    scf.for %i = %c0 to %c64 step %c8 {
      codir.codelet deps(%root : memref<64x32xf32>)
          params(%c8, %i : index, index)
          attributes {dep_modes = [#codir.access_mode<read>],
                      dep_storage_views = [#codir.storage_view<compute_block>],
                      distribution_kind = #codir.distribution_kind<blocked>,
                      iteration_topology = #codir.iteration_topology<owner_strip>,
                      logical_worker_slice = [8, 32],
                      pattern = #codir.pattern<uniform>,
                      tile_owner_dims = [0],
                      tile_shape = [8, 32]} {
      ^bb0(%arg0: memref<64x32xf32>, %step: index, %base: index):
        %inner_c0 = arith.constant 0 : index
        %loaded = memref.load %arg0[%base, %inner_c0] : memref<64x32xf32>
        codir.yield
      }
    }
    return
  }
}

// CHECK-LABEL: func.func @compute_block_intermediate_survives_across_codelets
// CHECK: %[[BGUID:.*]], %[[BPTR:.*]] = arts.db_alloc{{.*}}<block>
// CHECK-SAME: planPhysicalBlockShape = [8, 32]
// CHECK: arts.db_acquire[<inout>] (%[[BGUID]] : {{.*}}, %[[BPTR]] : {{.*}}) partitioning(<block>)
// CHECK: arts.edt <task> <internode>
// CHECK: arts.db_acquire[<in>] (%[[BGUID]] : {{.*}}, %[[BPTR]] : {{.*}}) partitioning(<block>)
// CHECK: arts.edt <task> <internode>
// CHECK-NOT: storageBridgeCopy
