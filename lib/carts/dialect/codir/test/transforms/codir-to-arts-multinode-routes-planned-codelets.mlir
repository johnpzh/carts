// RUN: %carts-compile %s --pass-pipeline='builtin.module(convert-sde-to-codir,verify-codir)' \
// RUN:   | %FileCheck %s --check-prefix=CODIR
// RUN: %carts-compile %s --pass-pipeline='builtin.module(convert-sde-to-codir,verify-codir,convert-codir-to-arts)' \
// RUN:   | %FileCheck %s --check-prefix=ARTS --implicit-check-not="arts.edt <task> <intranode>"

// CODIR remains machine-generic: it carries logical worker and physical owner
// slice metadata only. CODIR-to-ARTS is the first boundary that may consult the
// ARTS runtime topology and route planned codelet chunks across nodes.

module attributes {arts.runtime_total_nodes = 4 : i64, arts.runtime_total_workers = 256 : i64} {
  func.func @planned_owner_strip_routes_by_chunk(%A: memref<128x16xf32>, %B: memref<128x16xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    sde.cu_region <parallel> {
      sde.su_iterate (%c0) to (%c128) step (%c1) classification(<elementwise>) {
      ^bb0(%i: index):
        %v = memref.load %A[%i, %c0] : memref<128x16xf32>
        memref.store %v, %B[%i, %c0] : memref<128x16xf32>
        sde.yield
      } {iterationTopology = #sde.iteration_topology<owner_strip>,
         logicalWorkerSlice = [16, 16],
         pattern = #sde.pattern<uniform>,
         physicalBlockShape = [16, 16],
         physicalOwnerDims = [0]}
      sde.yield
    }
    return
  }
}

// CODIR-LABEL: func.func @planned_owner_strip_routes_by_chunk
// CODIR-NOT: arts.runtime_query
// CODIR-NOT: route(
// CODIR-NOT: <internode>
// CODIR: codir.codelet
// CODIR-SAME: iteration_topology = #codir.iteration_topology<owner_strip>
// CODIR-SAME: logical_worker_slice = [16, 16]
// CODIR-SAME: physical_block_shape = [16, 16]

// ARTS-LABEL: func.func @planned_owner_strip_routes_by_chunk
// ARTS: scf.for %[[IV:.*]] =
// ARTS: %[[REL:.*]] = arith.subi %[[IV]],
// ARTS: %[[ORD:.*]] = arith.divui %[[REL]],
// ARTS: %[[ORD_I32:.*]] = arith.index_cast %[[ORD]] : index to i32
// ARTS: %[[NODES:.*]] = arts.runtime_query <total_nodes> -> i32
// ARTS: %[[ROUTE:.*]] = arith.remui %[[ORD_I32]], %[[NODES]] : i32
// ARTS: arts.edt <task> <internode> route(%[[ROUTE]])
