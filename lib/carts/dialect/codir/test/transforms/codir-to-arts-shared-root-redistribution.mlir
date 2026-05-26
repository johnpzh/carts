// RUN: %carts-compile %s --arts-config %inputs_dir/arts_multinode_8x64.cfg \
// RUN:   --pass-pipeline='builtin.module(storage-planning,convert-codir-to-arts,verify-arts-objects-only)' \
// RUN:   | %FileCheck %s

module attributes {arts.runtime_total_nodes = 4 : i64, arts.runtime_total_workers = 256 : i64} {
  func.func @shared_root_block_writer_lowers_to_distributed_bridge() {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %root = sde.mu_alloc : memref<128xf32>
    scf.for %i = %c0 to %c128 step %c16 {
      codir.codelet deps(%root : memref<128xf32>)
          params(%c16, %i : index, index)
          attributes {dep_modes = [#codir.access_mode<readwrite>],
                      dep_storage_views = [#codir.storage_view<compute_block>],
                      distribution_kind = #codir.distribution_kind<blocked>,
                      iteration_topology = #codir.iteration_topology<owner_strip>,
                      logical_worker_slice = [16],
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

// CHECK-LABEL: func.func @shared_root_block_writer_lowers_to_distributed_bridge
// CHECK: arts.db_alloc
// CHECK-SAME: <coarse>
// CHECK: arts.db_alloc
// CHECK-SAME: <block>
// CHECK-SAME: storage_bridge = #arts.storage_bridge<host_whole_to_compute_block>
// CHECK: arts.db_acquire[<in>]
// CHECK-SAME: partitioning(<coarse>)
// CHECK: arts.db_acquire[<out>]
// CHECK-SAME: partitioning(<block>)
// CHECK: arts.edt <task> <internode>
// CHECK-SAME: storageBridgeCopy
// CHECK: arts.db_acquire[<inout>]
// CHECK-SAME: partitioning(<block>)
// CHECK: arts.edt <task> <internode>
// CHECK: arts.barrier
// CHECK: arts.db_acquire[<inout>]
// CHECK-SAME: partitioning(<coarse>)
// CHECK: arts.db_acquire[<in>]
// CHECK-SAME: partitioning(<block>)
// CHECK: arts.edt <task> <intranode>
// CHECK-SAME: storageBridgeCopy
// CHECK: arts.barrier
// CHECK: arts.db_acquire[<in>]
// CHECK-SAME: partitioning(<coarse>)
