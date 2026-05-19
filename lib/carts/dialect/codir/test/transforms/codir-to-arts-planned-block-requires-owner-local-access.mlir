// RUN: %carts-compile %s --pass-pipeline='builtin.module(verify-codir,convert-codir-to-arts,verify-arts-objects-only)' \
// RUN:   | %FileCheck %s --implicit-check-not="arts.edt <task> <internode>"

// Planned block storage is only legal when the codelet accesses the dependency
// through the owner dimension selected by the dispatch chunk. A cross-owner
// access must stay on the coarse local bridge until SDE/CODIR can materialize
// an explicit redistribution or replicated-read plan.

module attributes {arts.runtime_total_nodes = 4 : i64, arts.runtime_total_workers = 256 : i64} {
  func.func @cross_owner_arg_stays_local(%A: memref<?x?xf32>, %n: index) {
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
        %v = memref.load %a[%row, %i_arg] : memref<?x?xf32>
        memref.store %v, %a[%row, %i_arg] : memref<?x?xf32>
        codir.yield
      }
    }
    func.return
  }
}

// CHECK-LABEL: func.func @cross_owner_arg_stays_local
// CHECK: arts.db_alloc
// CHECK-SAME: <coarse>
// CHECK: arts.edt <task> <intranode>
// CHECK-SAME: planPhysicalBlockShape = [16, 16]
