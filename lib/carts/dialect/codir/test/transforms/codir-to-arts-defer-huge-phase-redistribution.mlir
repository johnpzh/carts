// RUN: %carts-compile %s --pass-pipeline='builtin.module(verify-codir,storage-planning,convert-codir-to-arts,verify-arts-objects-only)' \
// RUN:   --arts-config %inputs_dir/arts_multinode_4x16.cfg | %FileCheck %s \
// RUN:   --implicit-check-not='storage_bridge = #arts.storage_bridge<host_whole_to_compute_block>' \
// RUN:   --implicit-check-not='arts.edt <task> <internode>'

// A read-only host tensor whose codelet owner maps to a non-leading physical
// dimension can be redistributed into block DBs, but doing that for huge phase
// tensors is just a full-copy tax. Storage planning should leave it as a coarse
// local DB until the compiler has a true distributed reduction lowering.

module attributes {arts.runtime_total_nodes = 4 : i64, arts.runtime_total_workers = 64 : i64} {
  func.func @defer_huge_phase_redistribution() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %zero = arith.constant 0.0 : f32
    %A = memref.alloc() : memref<8x4194304x1xf32>
    memref.store %zero, %A[%c0, %c0, %c0] : memref<8x4194304x1xf32>
    scf.for %j = %c0 to %c16 step %c4 {
      codir.codelet deps(%A : memref<8x4194304x1xf32>)
          params(%j : index)
          attributes {dep_modes = [#codir.access_mode<read>],
                      dep_storage_views = [#codir.storage_view<compute_block>],
                      distribution_kind = #codir.distribution_kind<blocked>,
                      iteration_topology = #codir.iteration_topology<owner_strip>,
                      logical_worker_slice = [4],
                      pattern = #codir.pattern<elementwise_pipeline>,
                      tile_owner_dims = [0],
                      tile_shape = [4]} {
      ^bb0(%a: memref<8x4194304x1xf32>, %base: index):
        %inner_c0 = arith.constant 0 : index
        %inner_c1 = arith.constant 1 : index
        %inner_c4 = arith.constant 4 : index
        %inner_c16 = arith.constant 16 : index
        %end_raw = arith.addi %base, %inner_c4 : index
        %end = arith.minui %end_raw, %inner_c16 : index
        scf.for %jj = %base to %end step %inner_c1 {
          %v = memref.load %a[%inner_c0, %jj, %inner_c0] : memref<8x4194304x1xf32>
          func.call @use(%v) : (f32) -> ()
        }
        codir.yield
      }
    }
    memref.dealloc %A : memref<8x4194304x1xf32>
    return
  }

  func.func private @use(f32)
}

// CHECK-LABEL: func.func @defer_huge_phase_redistribution
// CHECK: arts.db_alloc
// CHECK-SAME: <coarse>
// CHECK: arts.edt <task> <intranode>
