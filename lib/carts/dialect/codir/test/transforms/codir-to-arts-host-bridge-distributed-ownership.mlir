// RUN: %carts-compile %s --pipeline post-db-refinement --arts-config %inputs_dir/arts_multinode_8x64.cfg --distributed-db | %FileCheck %s --implicit-check-not=unsupported_ptr_users

// Host-visible whole views remain coarse/local, but the bridge DB itself must
// not be rejected for distributed ownership merely because bridge movement is
// needed. Copy movement is represented as ARTS acquire/EDT work, so the block
// bridge can be distributed for the owner-slice compute tasks.

module attributes {arts.runtime_total_nodes = 8 : i64, arts.runtime_total_workers = 512 : i64} {
  func.func @write_only_bridge_distributed_after_copy_tasks() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
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
                      pattern = #codir.pattern<uniform>,
                      tile_owner_dims = [0],
                      tile_shape = [8, 4]} {
      ^bb0(%arg0: memref<18x4xf32>, %base: index):
        %inner_c0 = arith.constant 0 : index
        %inner_c1 = arith.constant 1 : index
        %inner_c4 = arith.constant 4 : index
        %inner_c8 = arith.constant 8 : index
        %inner_c18 = arith.constant 18 : index
        %value = arith.constant 2.000000e+00 : f32
        %end_raw = arith.addi %base, %inner_c8 : index
        %end = arith.minui %end_raw, %inner_c18 : index
        scf.for %row = %base to %end step %inner_c1 {
          scf.for %col = %inner_c0 to %inner_c4 step %inner_c1 {
            memref.store %value, %arg0[%row, %col] : memref<18x4xf32>
          }
        }
        codir.yield
      }
    }
    %result = memref.load %A[%c0, %c0] : memref<18x4xf32>
    func.call @use(%result) : (f32) -> ()
    memref.dealloc %A : memref<18x4xf32>
    return
  }

  func.func private @use(f32)
}

// CHECK-LABEL: func.func @write_only_bridge_distributed_after_copy_tasks
// CHECK: arts.db_alloc
// CHECK-SAME: <coarse>
// CHECK-SAME: arts.local_only
// CHECK: %[[BGUID:.*]], %[[BPTR:.*]] = arts.db_alloc
// CHECK-SAME: <block>
// CHECK-SAME: arts.storage_bridge = "host_whole_to_compute_block"
// CHECK-SAME: distributed
// CHECK: arts.edt <task> <internode>
// CHECK: arts.db_acquire[<inout>] (%{{.*}} : {{.*}}, %{{.*}} : {{.*}}) partitioning(<coarse>)
// CHECK: arts.db_acquire[<in>] (%[[BGUID]] : {{.*}}, %[[BPTR]] : {{.*}}) partitioning(<block>)
// CHECK: arts.edt <task> <intranode>
// CHECK: arts.barrier
