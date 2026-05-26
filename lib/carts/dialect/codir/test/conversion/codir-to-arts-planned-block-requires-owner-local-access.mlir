// RUN: %carts-compile %s --pass-pipeline='builtin.module(verify-codir,storage-planning,convert-codir-to-arts,verify-arts-objects-only)' \
// RUN:   | %FileCheck %s --implicit-check-not="arts.edt <task> <internode>"

// Planned block storage is only legal when the codelet accesses the dependency
// through a physical dimension selected by the dispatch chunk. Storage planning
// keeps a dependency that never uses the owner index on a coarse local view.

module attributes {arts.runtime_total_nodes = 4 : i64, arts.runtime_total_workers = 256 : i64} {
  func.func @non_owner_arg_stays_local(%A: memref<?x?xf32>, %n: index) {
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
      ^bb0(%a: memref<?x?xf32>, %i_arg: index):
        %row = arith.constant 0 : index
        %v = memref.load %a[%row, %row] : memref<?x?xf32>
        memref.store %v, %a[%row, %row] : memref<?x?xf32>
        codir.yield
      }
    }
    func.return
  }
}

// CHECK-LABEL: func.func @non_owner_arg_stays_local
// CHECK: arts.db_alloc
// CHECK-SAME: <coarse>
// CHECK: arts.edt <task> <intranode>
// CHECK-SAME: planPhysicalBlockShape = [16, 16]
