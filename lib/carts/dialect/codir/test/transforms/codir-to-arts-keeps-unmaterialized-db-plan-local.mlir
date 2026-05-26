// RUN: %carts-compile %s --pass-pipeline='builtin.module(verify-codir,storage-planning,convert-codir-to-arts,verify-arts-objects-only)' \
// RUN:   | %FileCheck %s

// A codelet may carry generic distribution intent before SDE/CODIR can
// materialize physical owner/block storage. CODIR-to-ARTS must not turn that
// into internode placement over coarse raw DBs. Once a planned owner-slice
// codelet has materialized block storage, the task can launch internode.

module attributes {arts.runtime_total_nodes = 4 : i64, arts.runtime_total_workers = 64 : i64} {
  func.func @unmaterialized_distributed_intent_stays_local(%A: memref<128xf64>,
                                                           %B: memref<128xf64>,
                                                           %i: index) {
    codir.codelet deps(%A, %B : memref<128xf64>, memref<128xf64>)
      params(%i : index)
      attributes {dep_modes = [#codir.access_mode<read>,
                               #codir.access_mode<write>],
                  distribution_kind = #codir.distribution_kind<blocked>,
                  pattern = #codir.pattern<uniform>} {
    ^bb0(%a: memref<128xf64>, %b: memref<128xf64>, %i_arg: index):
      %v = memref.load %a[%i_arg] : memref<128xf64>
      memref.store %v, %b[%i_arg] : memref<128xf64>
      codir.yield
    }
    func.return
  }

  func.func @planned_codelet_over_prebacked_coarse_db_uses_block_storage() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %route = arith.constant -1 : i32
    %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>, <coarse>]
      route(%route : i32) sizes[%c1] elementType(f64) elementSizes[%c128]
      : (memref<?xi64>, memref<?xmemref<?xf64>>)
    %A = arts.db_ref %ptr[%c0] : memref<?xmemref<?xf64>> -> memref<?xf64>
    scf.for %i = %c0 to %c128 step %c16 {
      codir.codelet deps(%A : memref<?xf64>)
        params(%i : index)
        attributes {dep_modes = [#codir.access_mode<readwrite>],
                    distribution_kind = #codir.distribution_kind<blocked>,
                    iteration_topology = #codir.iteration_topology<owner_strip>,
                    logical_worker_slice = [16],
                    pattern = #codir.pattern<uniform>,
                    tile_owner_dims = [0],
                    tile_shape = [16]} {
      ^bb0(%a: memref<?xf64>, %i_arg: index):
        %v = memref.load %a[%i_arg] : memref<?xf64>
        memref.store %v, %a[%i_arg] : memref<?xf64>
        codir.yield
      }
    }
    func.return
  }
}

// CHECK-LABEL: func.func @unmaterialized_distributed_intent_stays_local
// CHECK: arts.db_alloc
// CHECK-SAME: <coarse>
// CHECK: arts.edt <task> <intranode>
// CHECK-SAME: distribution_kind = #arts.distribution_kind<block>

// CHECK-LABEL: func.func @planned_codelet_over_prebacked_coarse_db_uses_block_storage
// CHECK: arts.db_alloc
// CHECK-SAME: <coarse>
// CHECK: arts.db_alloc
// CHECK-SAME: <block>
// CHECK-SAME: planPhysicalBlockShape = [16]
// CHECK: arts.edt <task> <internode>
// CHECK-SAME: planPhysicalBlockShape = [16]
