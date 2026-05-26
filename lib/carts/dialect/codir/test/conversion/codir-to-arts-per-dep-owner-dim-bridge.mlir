// RUN: %carts-compile %s --pass-pipeline='builtin.module(verify-codir,storage-planning,convert-codir-to-arts,verify-arts-objects-only)' \
// RUN:   --arts-config %inputs_dir/arts_multinode_4x16.cfg | %FileCheck %s

// A codelet can have one logical owner strip while different dependencies map
// that owner IV to different physical dimensions.  The rank-1 mean dependency
// is owned by dim 0, while the large rank-3 tensor dependency is owned by dim 1.
// Storage planning marks the host-whole tensor bridge explicitly; CODIR-to-ARTS
// then uses the per-dependency owner dimension for bridge DBs, block acquires,
// and local index rewrites.

module attributes {arts.runtime_total_nodes = 4 : i64, arts.runtime_total_workers = 64 : i64} {
  func.func @per_dep_owner_dim_bridge() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c2049 = arith.constant 2049 : index
    %zero = arith.constant 0.0 : f32
    %A = memref.alloc() : memref<8x2049x4xf32>
    %mean = memref.alloc() : memref<16xf32>
    memref.store %zero, %A[%c0, %c0, %c0] : memref<8x2049x4xf32>
    scf.for %j = %c0 to %c16 step %c4 {
      codir.codelet deps(%mean, %A : memref<16xf32>, memref<8x2049x4xf32>)
          params(%j : index)
          attributes {dep_modes = [#codir.access_mode<readwrite>, #codir.access_mode<read>],
                      dep_storage_views = [#codir.storage_view<compute_block>, #codir.storage_view<compute_block>],
                      distribution_kind = #codir.distribution_kind<blocked>,
                      in_place_safe,
                      iteration_topology = #codir.iteration_topology<owner_strip>,
                      logical_worker_slice = [4],
                      pattern = #codir.pattern<elementwise_pipeline>,
                      tile_owner_dims = [0],
                      tile_shape = [4]} {
      ^bb0(%m: memref<16xf32>, %a: memref<8x2049x4xf32>, %base: index):
        %inner_c0 = arith.constant 0 : index
        %inner_c1 = arith.constant 1 : index
        %inner_c4 = arith.constant 4 : index
        %inner_c8 = arith.constant 8 : index
        %inner_c16 = arith.constant 16 : index
        %inner_zero = arith.constant 0.0 : f32
        %end_raw = arith.addi %base, %inner_c4 : index
        %end = arith.minui %end_raw, %inner_c16 : index
        scf.for %jj = %base to %end step %inner_c1 {
          memref.store %inner_zero, %m[%jj] : memref<16xf32>
          scf.for %b = %inner_c0 to %inner_c8 step %inner_c1 {
            scf.for %k = %inner_c0 to %inner_c4 step %inner_c1 {
              %v = memref.load %a[%b, %jj, %k] : memref<8x2049x4xf32>
              %old = memref.load %m[%jj] : memref<16xf32>
              %next = arith.addf %old, %v : f32
              memref.store %next, %m[%jj] : memref<16xf32>
            }
          }
        }
        codir.yield
      }
    }
    %result = memref.load %mean[%c0] : memref<16xf32>
    func.call @use(%result) : (f32) -> ()
    memref.dealloc %mean : memref<16xf32>
    memref.dealloc %A : memref<8x2049x4xf32>
    return
  }

  func.func private @use(f32)
}

// CHECK-LABEL: func.func @per_dep_owner_dim_bridge
// CHECK: arts.db_alloc{{.*}}<block>
// CHECK-SAME: elementSizes[%c4
// CHECK-SAME: planOwnerDims = [0]
// CHECK-SAME: planPhysicalBlockShape = [4]
// CHECK: arts.db_alloc{{.*}}<block>
// CHECK-SAME: elementSizes[%c8{{.*}}, %c4{{.*}}, %c4
// CHECK-SAME: planOwnerDims = [1]
// CHECK-SAME: planPhysicalBlockShape = [4]
// CHECK: arts.db_acquire[<inout>]{{.*}}partitioning(<block>)
// CHECK: arts.db_acquire[<in>]{{.*}}partitioning(<block>)
// CHECK: arts.edt <task> <internode>
// CHECK-SAME: depPattern = #arts.dep_pattern<elementwise_pipeline>
// CHECK: arith.subi %{{.*}}, %{{.*}} : index
// CHECK: memref.load {{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32>
