// RUN: %carts-compile %s --pass-pipeline='builtin.module(verify-codir,storage-planning,convert-codir-to-arts)' \
// RUN:   --arts-config %inputs_dir/arts_multinode_8x64.cfg | %FileCheck %s --check-prefixes=CHECK,WRITE,WRITEFIRST,HOIST,SHARED,READONLY,PERSIST

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
    %A = memref.alloc() : memref<18x4xf32>
    %B = memref.alloc() : memref<18xf32>
    scf.for %rep = %c0 to %c2 step %c1 {
      scf.for %i = %c0 to %c18 step %c8 {
        codir.codelet deps(%A, %B : memref<18x4xf32>, memref<18xf32>) params(%i : index)
            attributes {dep_modes = [#codir.access_mode<read>, #codir.access_mode<write>],
                        dep_storage_views = [#codir.storage_view<compute_block>, #codir.storage_view<host_whole>],
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
    }
    %result = memref.load %A[%c0, %c0] : memref<18x4xf32>
    func.call @use(%result) : (f32) -> ()
    memref.dealloc %B : memref<18xf32>
    memref.dealloc %A : memref<18x4xf32>
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
}

// CHECK-LABEL: func.func @host_whole_to_compute_block_bridge
// CHECK: %[[HOST:.*]] = memref.alloc() : memref<18x4xf32>
// CHECK: arts.db_alloc
// CHECK-SAME: <block>
// CHECK-SAME: arts.storage_bridge = "host_whole_to_compute_block"
// CHECK-SAME: planPhysicalBlockShape = [8, 4]
// CHECK: scf.for
// CHECK: arts.db_ref
// CHECK: memref.load %[[HOST]]
// CHECK: memref.store
// CHECK: arts.edt <task> <internode>
// CHECK: arts.barrier
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.store {{.*}}%[[HOST]]
// CHECK: memref.load %[[HOST]]

// WRITE-LABEL: func.func @write_only_host_bridge_skips_copy_in
// WRITE: %[[HOST:.*]] = memref.alloc() : memref<18x4xf32>
// WRITE: arts.db_alloc
// WRITE-SAME: <block>
// WRITE-SAME: arts.storage_bridge = "host_whole_to_compute_block"
// WRITE-NOT: memref.load %[[HOST]]
// WRITE: arts.edt <task> <internode>
// WRITE: arts.barrier
// WRITE: memref.store {{.*}}%[[HOST]]

// WRITEFIRST-LABEL: func.func @write_first_shared_bridge_skips_initial_copy_in
// WRITEFIRST: %[[HOST:.*]] = memref.alloc() : memref<18x4xf32>
// WRITEFIRST-COUNT-1: arts.storage_bridge = "host_whole_to_compute_block"
// WRITEFIRST-NOT: memref.load %[[HOST]]
// WRITEFIRST: arts.edt <task> <internode>
// WRITEFIRST: arts.barrier
// WRITEFIRST: arts.edt <task> <internode>
// WRITEFIRST: arts.barrier
// WRITEFIRST: memref.store {{.*}}%[[HOST]]

// HOIST-LABEL: func.func @host_bridge_hoists_across_repetition
// HOIST: %[[HOST:.*]] = memref.alloc() : memref<18x4xf32>
// HOIST: arts.db_alloc
// HOIST-SAME: <block>
// HOIST-SAME: arts.storage_bridge = "host_whole_to_compute_block"
// HOIST: memref.load %[[HOST]]
// HOIST: scf.for
// HOIST: arts.edt <task> <internode>
// HOIST: arts.barrier
// HOIST: memref.store {{.*}}%[[HOST]]

// SHARED-LABEL: func.func @host_bridge_shared_across_phases
// SHARED: %[[HOST:.*]] = memref.alloc() : memref<18x4xf32>
// SHARED-COUNT-1: arts.storage_bridge = "host_whole_to_compute_block"
// SHARED: memref.load %[[HOST]]
// SHARED: scf.for
// SHARED: arts.edt <task> <internode>
// SHARED: arts.barrier
// SHARED: arts.edt <task> <internode>
// SHARED: arts.barrier
// SHARED: memref.store {{.*}}%[[HOST]]

// READONLY-LABEL: func.func @read_only_bridge_hoists_past_host_read
// READONLY: arts.db_alloc
// READONLY-SAME: <coarse>
// READONLY: %[[HOST:.*]] = arts.db_ref
// READONLY: arts.db_alloc{{.*}}arts.storage_bridge = "host_whole_to_compute_block"
// READONLY: memref.load %[[HOST]]
// READONLY: scf.for
// READONLY-NOT: arts.storage_bridge = "host_whole_to_compute_block"
// READONLY: arts.edt <task>
// READONLY: arts.edt <task>
// READONLY-NOT: arts.storage_bridge = "host_whole_to_compute_block"
// READONLY: memref.load %[[HOST]]

// PERSIST-LABEL: func.func @write_bridge_persists_across_read_only_host_phase
// PERSIST: %[[HOST:.*]] = arts.db_ref
// PERSIST-COUNT-1: arts.storage_bridge = "host_whole_to_compute_block"
// PERSIST: memref.load %[[HOST]]
// PERSIST: scf.for
// PERSIST: arts.edt <task> <internode>
// PERSIST: arts.barrier
// PERSIST: memref.store {{.*}}%[[HOST]]
// PERSIST: arts.edt <task> <intranode>
// PERSIST-NOT: memref.store {{.*}}%[[HOST]]
// PERSIST: arts.edt <task> <intranode>
// PERSIST: arts.edt <task> <internode>
// PERSIST: arts.barrier
// PERSIST: memref.store {{.*}}%[[HOST]]
