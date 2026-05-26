// RUN: %carts-compile %s --pass-pipeline='builtin.module(verify-codir,storage-planning,verify-codir)' \
// RUN:   | %FileCheck %s

// Storage planning owns the policy decision that used to be implicit in
// CODIR-to-ARTS: keep owner-local block deps as compute blocks, keep small
// read-only host-whole deps coarse to avoid broadcast bridge churn, mark
// affordable write/readwrite host-whole bridges as phase_redistributed, and
// leave huge redistribution candidates local when the bridge would be too large.

module {
  func.func @small_readonly_host_whole_dep_stays_host_whole() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %zero = arith.constant 0.0 : f32
    %A = memref.alloc() : memref<8x16x4xf32>
    memref.store %zero, %A[%c0, %c0, %c0] : memref<8x16x4xf32>
    scf.for %j = %c0 to %c16 step %c4 {
      codir.codelet deps(%A : memref<8x16x4xf32>)
          params(%j : index)
          attributes {dep_modes = [#codir.access_mode<read>],
                      dep_storage_views = [#codir.storage_view<compute_block>],
                      distribution_kind = #codir.distribution_kind<blocked>,
                      iteration_topology = #codir.iteration_topology<owner_strip>,
                      logical_worker_slice = [4],
                      pattern = #codir.pattern<elementwise_pipeline>,
                      tile_owner_dims = [0],
                      tile_shape = [4]} {
      ^bb0(%a: memref<8x16x4xf32>, %base: index):
        %inner_c0 = arith.constant 0 : index
        %inner_c1 = arith.constant 1 : index
        %inner_c4 = arith.constant 4 : index
        %inner_c16 = arith.constant 16 : index
        %end_raw = arith.addi %base, %inner_c4 : index
        %end = arith.minui %end_raw, %inner_c16 : index
        scf.for %jj = %base to %end step %inner_c1 {
          %v = memref.load %a[%inner_c0, %jj, %inner_c0] : memref<8x16x4xf32>
          func.call @use(%v) : (f32) -> ()
        }
        codir.yield
      }
    }
    memref.dealloc %A : memref<8x16x4xf32>
    return
  }

  func.func @huge_host_whole_dep_stays_local() {
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

  func.func @small_readwrite_host_whole_dep_gets_phase_plan() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %zero = arith.constant 0.0 : f32
    %A = memref.alloc() : memref<8x16x4xf32>
    memref.store %zero, %A[%c0, %c0, %c0] : memref<8x16x4xf32>
    scf.for %j = %c0 to %c16 step %c4 {
      codir.codelet deps(%A : memref<8x16x4xf32>)
          params(%j : index)
          attributes {dep_modes = [#codir.access_mode<readwrite>],
                      dep_storage_views = [#codir.storage_view<compute_block>],
                      distribution_kind = #codir.distribution_kind<blocked>,
                      iteration_topology = #codir.iteration_topology<owner_strip>,
                      logical_worker_slice = [4],
                      pattern = #codir.pattern<elementwise_pipeline>,
                      tile_owner_dims = [0],
                      tile_shape = [4]} {
      ^bb0(%a: memref<8x16x4xf32>, %base: index):
        %inner_c0 = arith.constant 0 : index
        %inner_c1 = arith.constant 1 : index
        %inner_c4 = arith.constant 4 : index
        %inner_c16 = arith.constant 16 : index
        %end_raw = arith.addi %base, %inner_c4 : index
        %end = arith.minui %end_raw, %inner_c16 : index
        scf.for %jj = %base to %end step %inner_c1 {
          %v = memref.load %a[%inner_c0, %jj, %inner_c0] : memref<8x16x4xf32>
          memref.store %v, %a[%inner_c0, %jj, %inner_c0] : memref<8x16x4xf32>
        }
        codir.yield
      }
    }
    memref.dealloc %A : memref<8x16x4xf32>
    return
  }

  func.func @missing_view_gets_explicit_phase_plan(%A: memref<?x?xf32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    scf.for %i = %c0 to %n step %c16 {
      codir.codelet deps(%A : memref<?x?xf32>)
          params(%i : index)
          attributes {dep_modes = [#codir.access_mode<readwrite>],
                      distribution_kind = #codir.distribution_kind<blocked>,
                      iteration_topology = #codir.iteration_topology<owner_strip>,
                      logical_worker_slice = [16, 16],
                      pattern = #codir.pattern<uniform>,
                      tile_owner_dims = [0],
                      tile_shape = [16, 16]} {
      ^bb0(%a: memref<?x?xf32>, %base: index):
        %col = arith.constant 0 : index
        %v = memref.load %a[%base, %col] : memref<?x?xf32>
        memref.store %v, %a[%base, %col] : memref<?x?xf32>
        codir.yield
      }
    }
    return
  }

  func.func @owner_local_body_subview_keeps_compute_block() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %A = memref.alloc() : memref<16x16xf32>
    scf.for %i = %c0 to %c16 step %c4 {
      codir.codelet deps(%A : memref<16x16xf32>)
          params(%i : index)
          attributes {dep_modes = [#codir.access_mode<readwrite>],
                      dep_storage_views = [#codir.storage_view<compute_block>],
                      distribution_kind = #codir.distribution_kind<blocked>,
                      iteration_topology = #codir.iteration_topology<owner_strip>,
                      logical_worker_slice = [4, 16],
                      pattern = #codir.pattern<uniform>,
                      tile_owner_dims = [0],
                      tile_shape = [4, 16]} {
      ^bb0(%a: memref<16x16xf32>, %base: index):
        %inner_c0 = arith.constant 0 : index
        %view = memref.subview %a[%base, %inner_c0] [1, 16] [1, 1]
          : memref<16x16xf32> to memref<1x16xf32, strided<[16, 1], offset: ?>>
        %v = memref.load %view[%inner_c0, %inner_c0]
          : memref<1x16xf32, strided<[16, 1], offset: ?>>
        memref.store %v, %view[%inner_c0, %inner_c0]
          : memref<1x16xf32, strided<[16, 1], offset: ?>>
        codir.yield
      }
    }
    memref.dealloc %A : memref<16x16xf32>
    return
  }

  func.func @owner_local_body_rank_reducing_subview_keeps_compute_block() {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %A = memref.alloc() : memref<16x16xf32>
    scf.for %i = %c0 to %c16 step %c4 {
      codir.codelet deps(%A : memref<16x16xf32>)
          params(%i : index)
          attributes {dep_modes = [#codir.access_mode<readwrite>],
                      dep_storage_views = [#codir.storage_view<compute_block>],
                      distribution_kind = #codir.distribution_kind<blocked>,
                      iteration_topology = #codir.iteration_topology<owner_strip>,
                      logical_worker_slice = [4, 16],
                      pattern = #codir.pattern<uniform>,
                      tile_owner_dims = [0],
                      tile_shape = [4, 16]} {
      ^bb0(%a: memref<16x16xf32>, %base: index):
        %inner_c0 = arith.constant 0 : index
        %view = memref.subview %a[%base, %inner_c0] [1, 16] [1, 1]
          : memref<16x16xf32> to memref<16xf32, strided<[1], offset: ?>>
        %v = memref.load %view[%inner_c0]
          : memref<16xf32, strided<[1], offset: ?>>
        memref.store %v, %view[%inner_c0]
          : memref<16xf32, strided<[1], offset: ?>>
        codir.yield
      }
    }
    memref.dealloc %A : memref<16x16xf32>
    return
  }

  func.func @owner_local_body_subindex_keeps_compute_block() {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %A = memref.alloc() : memref<16x16xf32>
    scf.for %i = %c0 to %c16 step %c4 {
      codir.codelet deps(%A : memref<16x16xf32>)
          params(%i : index)
          attributes {dep_modes = [#codir.access_mode<readwrite>],
                      dep_storage_views = [#codir.storage_view<compute_block>],
                      distribution_kind = #codir.distribution_kind<blocked>,
                      iteration_topology = #codir.iteration_topology<owner_strip>,
                      logical_worker_slice = [4, 16],
                      pattern = #codir.pattern<uniform>,
                      tile_owner_dims = [0],
                      tile_shape = [4, 16]} {
      ^bb0(%a: memref<16x16xf32>, %base: index):
        %inner_c0 = arith.constant 0 : index
        %row = polygeist.subindex %a[%base] () : memref<16x16xf32> -> memref<16xf32>
        %v = memref.load %row[%inner_c0] : memref<16xf32>
        memref.store %v, %row[%inner_c0] : memref<16xf32>
        codir.yield
      }
    }
    memref.dealloc %A : memref<16x16xf32>
    return
  }

  func.func @unrelated_rank_reducing_view_does_not_block_owner_local_dep() {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %A = memref.alloc() : memref<16x16xf32>
    scf.for %i = %c0 to %c16 step %c4 {
      codir.codelet deps(%A : memref<16x16xf32>)
          params(%i : index)
          attributes {dep_modes = [#codir.access_mode<readwrite>],
                      dep_storage_views = [#codir.storage_view<compute_block>],
                      distribution_kind = #codir.distribution_kind<blocked>,
                      iteration_topology = #codir.iteration_topology<owner_strip>,
                      logical_worker_slice = [4, 16],
                      pattern = #codir.pattern<uniform>,
                      tile_owner_dims = [0],
                      tile_shape = [4, 16]} {
      ^bb0(%a: memref<16x16xf32>, %base: index):
        %inner_c0 = arith.constant 0 : index
        %scratch = memref.alloca() : memref<1x16xf32>
        %scratch_row = memref.subview %scratch[%inner_c0, %inner_c0] [1, 16] [1, 1]
          : memref<1x16xf32> to memref<16xf32, strided<[1], offset: ?>>
        %sv = memref.load %scratch_row[%inner_c0]
          : memref<16xf32, strided<[1], offset: ?>>
        memref.store %sv, %scratch_row[%inner_c0]
          : memref<16xf32, strided<[1], offset: ?>>
        %v = memref.load %a[%base, %inner_c0] : memref<16x16xf32>
        memref.store %v, %a[%base, %inner_c0] : memref<16x16xf32>
        codir.yield
      }
    }
    memref.dealloc %A : memref<16x16xf32>
    return
  }

  func.func private @use(f32)
}

// CHECK-LABEL: func.func @small_readonly_host_whole_dep_stays_host_whole
// CHECK: codir.codelet
// CHECK-SAME: dep_storage_views = [#codir.storage_view<host_whole>]

// CHECK-LABEL: func.func @huge_host_whole_dep_stays_local
// CHECK: codir.codelet
// CHECK-SAME: dep_storage_views = [#codir.storage_view<host_whole>]

// CHECK-LABEL: func.func @small_readwrite_host_whole_dep_gets_phase_plan
// CHECK: codir.codelet
// CHECK-SAME: dep_storage_views = [#codir.storage_view<phase_redistributed>]

// CHECK-LABEL: func.func @missing_view_gets_explicit_phase_plan
// CHECK: codir.codelet
// CHECK-SAME: dep_storage_views = [#codir.storage_view<phase_redistributed>]

// CHECK-LABEL: func.func @owner_local_body_subview_keeps_compute_block
// CHECK: codir.codelet
// CHECK-SAME: dep_storage_views = [#codir.storage_view<compute_block>]

// CHECK-LABEL: func.func @owner_local_body_rank_reducing_subview_keeps_compute_block
// CHECK: codir.codelet
// CHECK-SAME: dep_storage_views = [#codir.storage_view<compute_block>]

// CHECK-LABEL: func.func @owner_local_body_subindex_keeps_compute_block
// CHECK: codir.codelet
// CHECK-SAME: dep_storage_views = [#codir.storage_view<compute_block>]

// CHECK-LABEL: func.func @unrelated_rank_reducing_view_does_not_block_owner_local_dep
// CHECK: codir.codelet
// CHECK-SAME: dep_storage_views = [#codir.storage_view<compute_block>]
