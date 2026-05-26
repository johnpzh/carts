// RUN: %carts-compile %s --pass-pipeline='builtin.module(verify-codir,storage-planning,convert-codir-to-arts,verify-arts-objects-only)' \
// RUN:   --arts-config %inputs_dir/arts_multinode_4x16.cfg | %FileCheck %s

module attributes {arts.runtime_total_nodes = 4 : i64, arts.runtime_total_workers = 64 : i64} {
  func.func @matmul_large_read_rhs_allows_internode() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c19 = arith.constant 19 : index
    %c4097 = arith.constant 4097 : index
    %zero = arith.constant 0.0 : f32
    %A = memref.alloc() : memref<4097x4097xf32>
    %B = memref.alloc() : memref<4097x4097xf32>
    %C = memref.alloc() : memref<4097x4097xf32>
    memref.store %zero, %A[%c0, %c0] : memref<4097x4097xf32>
    memref.store %zero, %B[%c0, %c0] : memref<4097x4097xf32>
    memref.store %zero, %C[%c0, %c0] : memref<4097x4097xf32>
    scf.for %i = %c0 to %c4097 step %c19 {
      codir.codelet deps(%C, %A, %B : memref<4097x4097xf32>, memref<4097x4097xf32>, memref<4097x4097xf32>)
          params(%c0, %c1, %c19, %c4097, %i : index, index, index, index, index)
          attributes {completion_barrier,
                      dep_modes = [#codir.access_mode<readwrite>, #codir.access_mode<read>, #codir.access_mode<read>],
                      dep_storage_views = [#codir.storage_view<compute_block>, #codir.storage_view<compute_block>, #codir.storage_view<host_whole>],
                      distribution_kind = #codir.distribution_kind<blocked>,
                      iteration_topology = #codir.iteration_topology<owner_strip>,
                      logical_worker_slice = [19, 4097],
                      pattern = #codir.pattern<matmul>,
                      tile_owner_dims = [0],
                      tile_shape = [19, 4097]} {
      ^bb0(%c: memref<4097x4097xf32>, %a: memref<4097x4097xf32>, %b: memref<4097x4097xf32>, %zero_idx: index, %one: index, %chunk: index, %n: index, %base: index):
        %owner_end_0 = arith.addi %base, %chunk : index
        %owner_end = arith.minui %owner_end_0, %n : index
        scf.for %tile = %base to %owner_end step %chunk {
          %tile_end_0 = arith.addi %tile, %chunk : index
          %tile_end = arith.minui %tile_end_0, %n : index
          scf.for %row = %tile to %tile_end step %one {
            scf.for %k = %zero_idx to %n step %one {
              scf.for %col = %zero_idx to %n step %one {
                %lhs = memref.load %a[%row, %k] : memref<4097x4097xf32>
                %rhs = memref.load %b[%k, %col] : memref<4097x4097xf32>
                %prod = arith.mulf %lhs, %rhs : f32
                %old = memref.load %c[%row, %col] : memref<4097x4097xf32>
                %next = arith.addf %old, %prod : f32
                memref.store %next, %c[%row, %col] : memref<4097x4097xf32>
              }
            }
          }
        }
        codir.yield
      }
    }
    %result = memref.load %C[%c0, %c0] : memref<4097x4097xf32>
    func.call @sink(%result) : (f32) -> ()
    memref.dealloc %C : memref<4097x4097xf32>
    memref.dealloc %B : memref<4097x4097xf32>
    memref.dealloc %A : memref<4097x4097xf32>
    return
  }

  func.func private @sink(f32)
}

// CHECK-LABEL: func.func @matmul_large_read_rhs_allows_internode
// CHECK-COUNT-2: arts.storage_bridge = "host_whole_to_compute_block"
// CHECK: arts.db_acquire[<in>] {{.*}} {replicatedRead}
// CHECK: arts.edt <task> <internode>
