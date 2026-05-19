// RUN: %carts-compile %s --pass-pipeline='builtin.module(verify-codir,storage-planning,convert-codir-to-arts,verify-arts-objects-only)' \
// RUN:   | %FileCheck %s

// A single-owner CODIR plan may partition a non-leading physical dimension.
// CODIR-to-ARTS must divide the owner base by that dimension's block size, not
// by tile_shape[0].  Otherwise tasks route to/acquire the wrong block whenever
// the owner dimension is not zero and the block shape is rectangular.

module attributes {arts.runtime_total_nodes = 4 : i64, arts.runtime_total_workers = 256 : i64} {
  func.func @nonleading_owner_dim_uses_matching_block_size(%A: memref<?x?xf32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    scf.for %j = %c0 to %n step %c16 {
      codir.codelet deps(%A : memref<?x?xf32>)
        params(%j : index)
        attributes {dep_modes = [#codir.access_mode<readwrite>],
                    distribution_kind = #codir.distribution_kind<blocked>,
                    iteration_topology = #codir.iteration_topology<owner_strip>,
                    logical_worker_slice = [8, 16],
                    pattern = #codir.pattern<uniform>,
                    tile_owner_dims = [1],
                    tile_shape = [8, 16]} {
      ^bb0(%a: memref<?x?xf32>, %j_arg: index):
        %row = arith.constant 0 : index
        %v = memref.load %a[%row, %j_arg] : memref<?x?xf32>
        memref.store %v, %a[%row, %j_arg] : memref<?x?xf32>
        codir.yield
      }
    }
    func.return
  }
}

// CHECK-LABEL: func.func @nonleading_owner_dim_uses_matching_block_size
// CHECK: scf.for
// CHECK-NOT: arith.divui %{{.*}}, %c8
// CHECK: %[[BLOCK_INDEX:.*]] = arith.divui %{{.*}}, %c16{{(_[0-9]+)?}} : index
// CHECK: arts.db_acquire
// CHECK-SAME: offsets[%[[BLOCK_INDEX]]]
// CHECK: arts.edt <task> <internode>
// CHECK-SAME: planOwnerDims = [1]
