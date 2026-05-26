// RUN: %carts-compile %s --pass-pipeline='builtin.module(verify-codir,storage-planning,convert-codir-to-arts)' \
// RUN:   --arts-config %inputs_dir/arts_multinode_8x64.cfg | %FileCheck %s

// Unclassified write-only compute-block codelets may update only part of the
// block. Seed the block DB from the host view before the write phase so
// untouched elements keep their source-program value.

// CHECK-LABEL: func.func @unclassified_write_only_bridge_copies_in
// CHECK: %[[HOST:.*]] = arts.db_ref
// CHECK: arts.db_alloc
// CHECK-SAME: <block>
// CHECK-SAME: storage_bridge = #arts.storage_bridge<host_whole_to_compute_block>
// CHECK: arts.db_acquire[<in>]
// CHECK-SAME: partitioning(<coarse>)
// CHECK: arts.db_acquire[<out>]
// CHECK-SAME: partitioning(<block>)
// CHECK: arts.edt <task> <internode>
// CHECK-SAME: storageBridgeCopy
// CHECK: arts.edt <task> <internode>
// CHECK: arts.barrier
// CHECK: arts.db_acquire[<inout>]
// CHECK-SAME: partitioning(<coarse>)
// CHECK: arts.db_acquire[<in>]
// CHECK-SAME: partitioning(<block>)
// CHECK: arts.edt <task> <intranode>
// CHECK-SAME: storageBridgeCopy
// CHECK: arts.barrier
// CHECK: memref.load %[[HOST]]

module attributes {arts.runtime_total_nodes = 8 : i64, arts.runtime_total_workers = 512 : i64} {
  func.func private @use(f32)

  func.func @unclassified_write_only_bridge_copies_in() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c18 = arith.constant 18 : index
    %A = memref.alloc() : memref<18x4xf32>
    scf.for %i = %c0 to %c18 step %c8 {
      codir.codelet deps(%A : memref<18x4xf32>) params(%i : index)
          attributes {dep_modes = [#codir.access_mode<write>],
                      dep_storage_views = [#codir.storage_view<compute_block>],
                      distribution_kind = #codir.distribution_kind<blocked>,
                      iteration_topology = #codir.iteration_topology<owner_strip>,
                      logical_worker_slice = [8, 4],
                      tile_owner_dims = [0],
                      tile_shape = [8, 4]} {
      ^bb0(%arg0: memref<18x4xf32>, %base: index):
        %inner_c0 = arith.constant 0 : index
        %inner_cst = arith.constant 2.000000e+00 : f32
        memref.store %inner_cst, %arg0[%base, %inner_c0] : memref<18x4xf32>
        codir.yield
      }
    }
    %result = memref.load %A[%c0, %c0] : memref<18x4xf32>
    func.call @use(%result) : (f32) -> ()
    memref.dealloc %A : memref<18x4xf32>
    return
  }
}
