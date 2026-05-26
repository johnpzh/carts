// RUN: %carts-compile %s --pass-pipeline='builtin.module(verify-codir,storage-planning,convert-codir-to-arts)' \
// RUN:   --arts-config %inputs_dir/arts_multinode_8x64.cfg | %FileCheck %s --check-prefixes=CHECK,WRITE,WRITEFIRST,HOIST,SHARED,READONLY,PERSIST,INCOMPAT

// Storage planning turns an owner-local compute_block request into an explicit
// phase_redistributed contract when host code still needs the original whole
// view. CODIR-to-ARTS then inserts bridge copies instead of replacing the host
// allocation with block zero. Write-only owner-slice bridges do not need to
// seed the block DB from stale host contents before the EDTs run. Repeated
// owner-slice waves with no host-whole uses inside the repetition keep the
// bridge DB live across repetitions and copy back once after the repetition.

module attributes {arts.runtime_total_nodes = 8 : i64, arts.runtime_total_workers = 512 : i64} {
  func.func @host_whole_to_compute_block_bridge() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c18 = arith.constant 18 : index
    %cst = arith.constant 1.000000e+00 : f32
    %A = memref.alloc() : memref<18x4xf32>
    memref.store %cst, %A[%c0, %c0] : memref<18x4xf32>
    scf.for %i = %c0 to %c18 step %c8 {
      codir.codelet deps(%A : memref<18x4xf32>) params(%i : index)
          attributes {dep_modes = [#codir.access_mode<readwrite>],
                      dep_storage_views = [#codir.storage_view<compute_block>],
                      distribution_kind = #codir.distribution_kind<blocked>,
                      iteration_topology = #codir.iteration_topology<owner_strip>,
                      logical_worker_slice = [8, 4],
                      pattern = #codir.pattern<uniform>,
                      tile_owner_dims = [0],
                      tile_shape = [8, 4]} {
      ^bb0(%arg0: memref<18x4xf32>, %base: index):
        %inner_c0 = arith.constant 0 : index
        %value = memref.load %arg0[%base, %inner_c0] : memref<18x4xf32>
        memref.store %value, %arg0[%base, %inner_c0] : memref<18x4xf32>
        codir.yield
      }
    }
    %result = memref.load %A[%c0, %c0] : memref<18x4xf32>
    func.call @use(%result) : (f32) -> ()
    memref.dealloc %A : memref<18x4xf32>
    return
  }

  func.func private @use(f32)

  func.func @write_only_host_bridge_skips_copy_in() {
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
        %inner_cst = arith.constant 2.000000e+00 : f32
        %end_raw = arith.addi %base, %inner_c8 : index
        %end = arith.minui %end_raw, %inner_c18 : index
        scf.for %row = %base to %end step %inner_c1 {
          scf.for %col = %inner_c0 to %inner_c4 step %inner_c1 {
            memref.store %inner_cst, %arg0[%row, %col] : memref<18x4xf32>
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

  func.func @write_first_shared_bridge_skips_initial_copy_in() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c18 = arith.constant 18 : index
    %A = memref.alloc() : memref<18x4xf32>
    scf.for %rep = %c0 to %c2 step %c1 {
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
          %inner_cst = arith.constant 2.000000e+00 : f32
          %end_raw = arith.addi %base, %inner_c8 : index
          %end = arith.minui %end_raw, %inner_c18 : index
          scf.for %row = %base to %end step %inner_c1 {
            scf.for %col = %inner_c0 to %inner_c4 step %inner_c1 {
              memref.store %inner_cst, %arg0[%row, %col] : memref<18x4xf32>
            }
          }
          codir.yield
        }
      }
      scf.for %j = %c0 to %c18 step %c8 {
        codir.codelet deps(%A : memref<18x4xf32>) params(%j : index)
            attributes {dep_modes = [#codir.access_mode<readwrite>],
                        dep_storage_views = [#codir.storage_view<compute_block>],
                        distribution_kind = #codir.distribution_kind<blocked>,
                        iteration_topology = #codir.iteration_topology<owner_strip>,
                        logical_worker_slice = [8, 4],
                        pattern = #codir.pattern<uniform>,
                        tile_owner_dims = [0],
                        tile_shape = [8, 4]} {
        ^bb0(%arg0: memref<18x4xf32>, %base: index):
          %inner_c0 = arith.constant 0 : index
          %inner_cst = arith.constant 1.000000e+00 : f32
          %value = memref.load %arg0[%base, %inner_c0] : memref<18x4xf32>
          %next = arith.addf %value, %inner_cst : f32
          memref.store %next, %arg0[%base, %inner_c0] : memref<18x4xf32>
          codir.yield
        }
      }
    }
    %result = memref.load %A[%c0, %c0] : memref<18x4xf32>
    func.call @use(%result) : (f32) -> ()
    memref.dealloc %A : memref<18x4xf32>
    return
  }

  func.func @host_bridge_hoists_across_repetition() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c18 = arith.constant 18 : index
    %A = memref.alloc() : memref<18x4xf32>
    scf.for %rep = %c0 to %c2 step %c1 {
      scf.for %i = %c0 to %c18 step %c8 {
        codir.codelet deps(%A : memref<18x4xf32>) params(%i : index)
            attributes {dep_modes = [#codir.access_mode<readwrite>],
                        dep_storage_views = [#codir.storage_view<compute_block>],
                        distribution_kind = #codir.distribution_kind<blocked>,
                        iteration_topology = #codir.iteration_topology<owner_strip>,
                        logical_worker_slice = [8, 4],
                        pattern = #codir.pattern<uniform>,
                        tile_owner_dims = [0],
                        tile_shape = [8, 4]} {
        ^bb0(%arg0: memref<18x4xf32>, %base: index):
          %inner_c0 = arith.constant 0 : index
          %value = memref.load %arg0[%base, %inner_c0] : memref<18x4xf32>
          memref.store %value, %arg0[%base, %inner_c0] : memref<18x4xf32>
          codir.yield
        }
      }
    }
    %result = memref.load %A[%c0, %c0] : memref<18x4xf32>
    func.call @use(%result) : (f32) -> ()
    memref.dealloc %A : memref<18x4xf32>
    return
  }

  func.func @host_bridge_shared_across_phases() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c18 = arith.constant 18 : index
    %A = memref.alloc() : memref<18x4xf32>
    scf.for %rep = %c0 to %c2 step %c1 {
      scf.for %i = %c0 to %c18 step %c8 {
        codir.codelet deps(%A : memref<18x4xf32>) params(%i : index)
            attributes {dep_modes = [#codir.access_mode<readwrite>],
                        dep_storage_views = [#codir.storage_view<compute_block>],
                        distribution_kind = #codir.distribution_kind<blocked>,
                        iteration_topology = #codir.iteration_topology<owner_strip>,
                        logical_worker_slice = [8, 4],
                        pattern = #codir.pattern<uniform>,
                        tile_owner_dims = [0],
                        tile_shape = [8, 4]} {
        ^bb0(%arg0: memref<18x4xf32>, %base: index):
          %inner_c0 = arith.constant 0 : index
          %value = memref.load %arg0[%base, %inner_c0] : memref<18x4xf32>
          memref.store %value, %arg0[%base, %inner_c0] : memref<18x4xf32>
          codir.yield
        }
      }
      scf.for %j = %c0 to %c18 step %c8 {
        codir.codelet deps(%A : memref<18x4xf32>) params(%j : index)
            attributes {dep_modes = [#codir.access_mode<readwrite>],
                        dep_storage_views = [#codir.storage_view<compute_block>],
                        distribution_kind = #codir.distribution_kind<blocked>,
                        iteration_topology = #codir.iteration_topology<owner_strip>,
                        logical_worker_slice = [8, 4],
                        pattern = #codir.pattern<uniform>,
                        tile_owner_dims = [0],
                        tile_shape = [8, 4]} {
        ^bb0(%arg0: memref<18x4xf32>, %base: index):
          %inner_c0 = arith.constant 0 : index
          %inner_cst = arith.constant 1.000000e+00 : f32
          %value = memref.load %arg0[%base, %inner_c0] : memref<18x4xf32>
          %next = arith.addf %value, %inner_cst : f32
          memref.store %next, %arg0[%base, %inner_c0] : memref<18x4xf32>
          codir.yield
        }
      }
    }
    %result = memref.load %A[%c0, %c0] : memref<18x4xf32>
    func.call @use(%result) : (f32) -> ()
    memref.dealloc %A : memref<18x4xf32>
    return
  }

  func.func @read_only_bridge_hoists_past_host_read() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c18 = arith.constant 18 : index
    %A = memref.alloc() : memref<131072x4xf32>
    %B = memref.alloc() : memref<18xf32>
    scf.for %rep = %c0 to %c2 step %c1 {
      scf.for %i = %c0 to %c18 step %c8 {
        codir.codelet deps(%A, %B : memref<131072x4xf32>, memref<18xf32>) params(%i : index)
            attributes {dep_modes = [#codir.access_mode<read>, #codir.access_mode<write>],
                        dep_storage_views = [#codir.storage_view<compute_block>, #codir.storage_view<host_whole>],
                        distribution_kind = #codir.distribution_kind<blocked>,
                        iteration_topology = #codir.iteration_topology<owner_strip>,
                        logical_worker_slice = [8, 4],
                        pattern = #codir.pattern<uniform>,
                        tile_owner_dims = [0],
                        tile_shape = [8, 4]} {
        ^bb0(%arg0: memref<131072x4xf32>, %arg1: memref<18xf32>, %base: index):
          %inner_c0 = arith.constant 0 : index
          %value = memref.load %arg0[%base, %inner_c0] : memref<131072x4xf32>
          memref.store %value, %arg1[%base] : memref<18xf32>
          codir.yield
        }
      }
      scf.for %j = %c0 to %c18 step %c8 {
        codir.codelet deps(%A, %B : memref<131072x4xf32>, memref<18xf32>) params(%j : index)
            attributes {dep_modes = [#codir.access_mode<read>, #codir.access_mode<write>],
                        dep_storage_views = [#codir.storage_view<host_whole>, #codir.storage_view<host_whole>],
                        distribution_kind = #codir.distribution_kind<blocked>,
                        iteration_topology = #codir.iteration_topology<owner_strip>,
                        logical_worker_slice = [8, 4],
                        pattern = #codir.pattern<uniform>,
                        tile_owner_dims = [0],
                        tile_shape = [8, 4]} {
        ^bb0(%arg0: memref<131072x4xf32>, %arg1: memref<18xf32>, %base: index):
          %inner_c0 = arith.constant 0 : index
          %value = memref.load %arg0[%base, %inner_c0] : memref<131072x4xf32>
          memref.store %value, %arg1[%base] : memref<18xf32>
          codir.yield
        }
      }
    }
    %result = memref.load %A[%c0, %c0] : memref<131072x4xf32>
    func.call @use(%result) : (f32) -> ()
    memref.dealloc %B : memref<18xf32>
    memref.dealloc %A : memref<131072x4xf32>
    return
  }

  func.func @write_bridge_persists_across_read_only_host_phase() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c18 = arith.constant 18 : index
    %A = memref.alloc() : memref<18x4xf32>
    %B = memref.alloc() : memref<18xf32>
    %C = memref.alloc() : memref<18xf32>
    scf.for %rep = %c0 to %c2 step %c1 {
      scf.for %i = %c0 to %c18 step %c8 {
        codir.codelet deps(%A : memref<18x4xf32>) params(%i : index)
            attributes {dep_modes = [#codir.access_mode<readwrite>],
                        dep_storage_views = [#codir.storage_view<compute_block>],
                        distribution_kind = #codir.distribution_kind<blocked>,
                        iteration_topology = #codir.iteration_topology<owner_strip>,
                        logical_worker_slice = [8, 4],
                        pattern = #codir.pattern<uniform>,
                        tile_owner_dims = [0],
                        tile_shape = [8, 4]} {
        ^bb0(%arg0: memref<18x4xf32>, %base: index):
          %inner_c0 = arith.constant 0 : index
          %inner_cst = arith.constant 1.000000e+00 : f32
          memref.store %inner_cst, %arg0[%base, %inner_c0] : memref<18x4xf32>
          codir.yield
        }
      }
      scf.for %j = %c0 to %c18 step %c8 {
        codir.codelet deps(%A, %B : memref<18x4xf32>, memref<18xf32>) params(%j : index)
            attributes {dep_modes = [#codir.access_mode<read>, #codir.access_mode<write>],
                        dep_storage_views = [#codir.storage_view<host_whole>, #codir.storage_view<host_whole>],
                        distribution_kind = #codir.distribution_kind<blocked>,
                        iteration_topology = #codir.iteration_topology<owner_strip>,
                        logical_worker_slice = [8, 4],
                        pattern = #codir.pattern<uniform>,
                        tile_owner_dims = [0],
                        tile_shape = [8, 4]} {
        ^bb0(%arg0: memref<18x4xf32>, %arg1: memref<18xf32>, %base: index):
          %inner_c0 = arith.constant 0 : index
          %value = memref.load %arg0[%base, %inner_c0] : memref<18x4xf32>
          memref.store %value, %arg1[%base] : memref<18xf32>
          codir.yield
        }
      }
      scf.for %h = %c0 to %c18 step %c8 {
        codir.codelet deps(%A, %C : memref<18x4xf32>, memref<18xf32>) params(%h : index)
            attributes {dep_modes = [#codir.access_mode<read>, #codir.access_mode<write>],
                        dep_storage_views = [#codir.storage_view<host_whole>, #codir.storage_view<host_whole>],
                        distribution_kind = #codir.distribution_kind<blocked>,
                        iteration_topology = #codir.iteration_topology<owner_strip>,
                        logical_worker_slice = [8, 4],
                        pattern = #codir.pattern<uniform>,
                        tile_owner_dims = [0],
                        tile_shape = [8, 4]} {
        ^bb0(%arg0: memref<18x4xf32>, %arg1: memref<18xf32>, %base: index):
          %inner_c0 = arith.constant 0 : index
          %value = memref.load %arg0[%base, %inner_c0] : memref<18x4xf32>
          memref.store %value, %arg1[%base] : memref<18xf32>
          codir.yield
        }
      }
      scf.for %k = %c0 to %c18 step %c8 {
        codir.codelet deps(%A : memref<18x4xf32>) params(%k : index)
            attributes {dep_modes = [#codir.access_mode<readwrite>],
                        dep_storage_views = [#codir.storage_view<compute_block>],
                        distribution_kind = #codir.distribution_kind<blocked>,
                        iteration_topology = #codir.iteration_topology<owner_strip>,
                        logical_worker_slice = [8, 4],
                        pattern = #codir.pattern<uniform>,
                        tile_owner_dims = [0],
                        tile_shape = [8, 4]} {
        ^bb0(%arg0: memref<18x4xf32>, %base: index):
          %inner_c0 = arith.constant 0 : index
          %inner_cst = arith.constant 1.000000e+00 : f32
          %value = memref.load %arg0[%base, %inner_c0] : memref<18x4xf32>
          %next = arith.addf %value, %inner_cst : f32
          memref.store %next, %arg0[%base, %inner_c0] : memref<18x4xf32>
          codir.yield
        }
      }
    }
    %result = memref.load %A[%c0, %c0] : memref<18x4xf32>
    func.call @use(%result) : (f32) -> ()
    memref.dealloc %C : memref<18xf32>
    memref.dealloc %B : memref<18xf32>
    memref.dealloc %A : memref<18x4xf32>
    return
  }

  func.func @incompatible_bridge_plan_copyin_stays_inside_repetition() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c18 = arith.constant 18 : index
    %A = memref.alloc() : memref<131072xf32>
    scf.for %rep = %c0 to %c2 step %c1 {
      scf.for %i = %c0 to %c18 step %c8 {
        codir.codelet deps(%A : memref<131072xf32>) params(%i : index)
            attributes {dep_modes = [#codir.access_mode<readwrite>],
                        dep_storage_views = [#codir.storage_view<compute_block>],
                        distribution_kind = #codir.distribution_kind<blocked>,
                        iteration_topology = #codir.iteration_topology<owner_strip>,
                        logical_worker_slice = [8],
                        pattern = #codir.pattern<uniform>,
                        tile_owner_dims = [0],
                        tile_shape = [8]} {
        ^bb0(%arg0: memref<131072xf32>, %base: index):
          %inner_c1 = arith.constant 1 : index
          %inner_c8 = arith.constant 8 : index
          %inner_c18 = arith.constant 18 : index
          %inner_cst = arith.constant 1.000000e+00 : f32
          %end_raw = arith.addi %base, %inner_c8 : index
          %end = arith.minui %end_raw, %inner_c18 : index
          scf.for %row = %base to %end step %inner_c1 {
            memref.store %inner_cst, %arg0[%row] : memref<131072xf32>
          }
          codir.yield
        }
      }
      scf.for %j = %c0 to %c18 step %c18 {
        codir.codelet deps(%A : memref<131072xf32>) params(%j : index)
            attributes {dep_modes = [#codir.access_mode<read>],
                        dep_storage_views = [#codir.storage_view<compute_block>],
                        distribution_kind = #codir.distribution_kind<blocked>,
                        iteration_topology = #codir.iteration_topology<owner_strip>,
                        logical_worker_slice = [18],
                        pattern = #codir.pattern<uniform>,
                        tile_owner_dims = [0],
                        tile_shape = [18]} {
        ^bb0(%arg0: memref<131072xf32>, %base: index):
          %value = memref.load %arg0[%base] : memref<131072xf32>
          codir.yield
        }
      }
    }
    %result = memref.load %A[%c0] : memref<131072xf32>
    func.call @use(%result) : (f32) -> ()
    memref.dealloc %A : memref<131072xf32>
    return
  }
}

// CHECK-LABEL: func.func @host_whole_to_compute_block_bridge
// CHECK: %[[HOST:.*]] = arts.db_ref
// CHECK: arts.db_alloc
// CHECK-SAME: <block>
// CHECK-SAME: arts.storage_bridge = "host_whole_to_compute_block"
// CHECK-SAME: planPhysicalBlockShape = [8, 4]
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

// WRITE-LABEL: func.func @write_only_host_bridge_skips_copy_in
// WRITE: %[[HOST:.*]] = arts.db_ref
// WRITE: arts.db_alloc
// WRITE-SAME: <block>
// WRITE-SAME: arts.storage_bridge = "host_whole_to_compute_block"
// WRITE-NOT: arts.db_acquire[<in>]
// WRITE: arts.edt <task> <internode>
// WRITE: arts.barrier
// WRITE: arts.db_acquire[<inout>]
// WRITE-SAME: partitioning(<coarse>)
// WRITE: arts.edt <task> <intranode>
// WRITE-SAME: storageBridgeCopy
// WRITE: arts.barrier
// WRITE: memref.load %[[HOST]]

// WRITEFIRST-LABEL: func.func @write_first_shared_bridge_skips_initial_copy_in
// WRITEFIRST: %[[HOST:.*]] = arts.db_ref
// WRITEFIRST-COUNT-1: arts.storage_bridge = "host_whole_to_compute_block"
// WRITEFIRST-NOT: arts.db_acquire[<in>]
// WRITEFIRST: arts.edt <task> <internode>
// WRITEFIRST: arts.barrier
// WRITEFIRST: arts.edt <task> <internode>
// WRITEFIRST: arts.barrier
// WRITEFIRST: arts.edt <task> <intranode>
// WRITEFIRST-SAME: storageBridgeCopy
// WRITEFIRST: arts.barrier
// WRITEFIRST: memref.load %[[HOST]]

// HOIST-LABEL: func.func @host_bridge_hoists_across_repetition
// HOIST: %[[HOST:.*]] = arts.db_ref
// HOIST: arts.db_alloc
// HOIST-SAME: <block>
// HOIST-SAME: arts.storage_bridge = "host_whole_to_compute_block"
// HOIST: arts.edt <task> <internode>
// HOIST-SAME: storageBridgeCopy
// HOIST: scf.for
// HOIST: arts.edt <task> <internode>
// HOIST: arts.barrier
// HOIST: arts.edt <task> <intranode>
// HOIST-SAME: storageBridgeCopy
// HOIST: arts.barrier
// HOIST: memref.load %[[HOST]]

// SHARED-LABEL: func.func @host_bridge_shared_across_phases
// SHARED: %[[HOST:.*]] = arts.db_ref
// SHARED-COUNT-1: arts.storage_bridge = "host_whole_to_compute_block"
// SHARED: arts.edt <task> <internode>
// SHARED-SAME: storageBridgeCopy
// SHARED: scf.for
// SHARED: arts.edt <task> <internode>
// SHARED: arts.barrier
// SHARED: arts.edt <task> <internode>
// SHARED: arts.barrier
// SHARED: arts.edt <task> <intranode>
// SHARED-SAME: storageBridgeCopy
// SHARED: arts.barrier
// SHARED: memref.load %[[HOST]]

// READONLY-LABEL: func.func @read_only_bridge_hoists_past_host_read
// READONLY: arts.db_alloc
// READONLY-SAME: <coarse>
// READONLY: %[[HOST:.*]] = arts.db_ref
// READONLY: arts.db_alloc{{.*}}arts.storage_bridge = "host_whole_to_compute_block"
// READONLY: arts.db_acquire[<in>]
// READONLY-SAME: partitioning(<coarse>)
// READONLY: arts.edt <task> <internode>
// READONLY-SAME: storageBridgeCopy
// READONLY: scf.for
// READONLY-NOT: arts.storage_bridge = "host_whole_to_compute_block"
// READONLY: arts.edt <task>
// READONLY: arts.edt <task>
// READONLY-NOT: arts.storage_bridge = "host_whole_to_compute_block"
// READONLY: memref.load %[[HOST]]

// PERSIST-LABEL: func.func @write_bridge_persists_across_read_only_host_phase
// PERSIST: %[[HOST:.*]] = arts.db_ref
// PERSIST-COUNT-1: arts.storage_bridge = "host_whole_to_compute_block"
// PERSIST: arts.edt <task> <internode>
// PERSIST-SAME: storageBridgeCopy
// PERSIST: scf.for
// PERSIST: arts.edt <task> <internode>
// PERSIST: arts.barrier
// PERSIST: arts.edt <task> <intranode>
// PERSIST-SAME: storageBridgeCopy
// PERSIST: arts.barrier
// PERSIST: arts.edt <task> <intranode>
// PERSIST: arts.edt <task> <intranode>
// PERSIST: arts.edt <task> <internode>
// PERSIST-SAME: storageBridgeCopy
// PERSIST: arts.edt <task> <internode>
// PERSIST: arts.barrier
// PERSIST: arts.edt <task> <intranode>
// PERSIST-SAME: storageBridgeCopy
// PERSIST: arts.barrier
// PERSIST: memref.load %[[HOST]]

// INCOMPAT-LABEL: func.func @incompatible_bridge_plan_copyin_stays_inside_repetition
// INCOMPAT: scf.for
// INCOMPAT: arts.edt <task> <internode>
// INCOMPAT: arts.barrier
// INCOMPAT: arts.edt <task> <intranode>
// INCOMPAT-SAME: storageBridgeCopy
// INCOMPAT: arts.barrier
// INCOMPAT: arts.db_alloc
// INCOMPAT-SAME: arts.storage_bridge = "host_whole_to_compute_block"
// INCOMPAT: arts.db_acquire[<in>]
// INCOMPAT-SAME: partitioning(<coarse>)
// INCOMPAT: arts.db_acquire[<out>]
// INCOMPAT-SAME: partitioning(<block>)
// INCOMPAT: arts.edt <task> <internode>
// INCOMPAT-SAME: storageBridgeCopy
// INCOMPAT: arts.edt <task> <internode>
