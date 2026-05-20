// RUN: %carts-compile %s --pass-pipeline='builtin.module(verify-codir,storage-planning,convert-codir-to-arts,verify-arts-objects-only)' \
// RUN:   | %FileCheck %s

// Dynamic function arguments still have concrete runtime extents available via
// memref.dim. Storage planning marks the function argument as a phase bridge,
// and CODIR-to-ARTS must preserve those extents when materializing both the
// host-whole DB wrapper and the physical block bridge.

module attributes {arts.runtime_total_nodes = 4 : i64, arts.runtime_total_workers = 256 : i64} {
  func.func @dynamic_arg_routes_with_block_storage(%A: memref<?x?xf32>, %n: index) {
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
        %col = arith.constant 0 : index
        %v = memref.load %a[%i_arg, %col] : memref<?x?xf32>
        memref.store %v, %a[%i_arg, %col] : memref<?x?xf32>
        codir.yield
      }
    }
    func.return
  }
}

// CHECK-LABEL: func.func @dynamic_arg_routes_with_block_storage
// CHECK: %[[D0:.*]] = memref.dim %arg0, %c0 : memref<?x?xf32>
// CHECK: %[[D1:.*]] = memref.dim %arg0, %c1 : memref<?x?xf32>
// CHECK: arts.db_alloc
// CHECK-SAME: <coarse>
// CHECK-SAME: elementSizes[%[[D0]], %[[D1]]]
// CHECK: arts.db_alloc
// CHECK-SAME: <block>
// CHECK-SAME: elementSizes[%c16{{(_[0-9]+)?}}, %[[D1]]]
// CHECK: arts.edt <task> <intranode>
// CHECK: arts.edt <task> <internode>
// CHECK-SAME: planPhysicalBlockShape = [16, 16]
// CHECK: %[[PAYLOAD:.*]] = arts.db_ref
// CHECK: %[[COL:.*]] = arith.constant 0 : index
// CHECK: %[[LOCAL_ROW:.*]] = arith.constant 0 : index
// CHECK: memref.load %[[PAYLOAD]][%[[LOCAL_ROW]], %[[COL]]]
