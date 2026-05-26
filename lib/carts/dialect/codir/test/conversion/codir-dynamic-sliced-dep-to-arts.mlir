// RUN: %carts-compile %s --pass-pipeline='builtin.module(verify-codir,convert-codir-to-arts)' | %FileCheck %s --implicit-check-not=codir.codelet

module {
  func.func @dynamic_sliced_dep_remaps_edt_params(%n: index, %off: index,
                                                  %i: index) {
    %root = sde.mu_alloc(%n) : memref<?xi32>
    %view = memref.subview %root[%off] [%n] [1]
      : memref<?xi32> to memref<?xi32, strided<[1], offset: ?>>

    codir.codelet deps(%view : memref<?xi32, strided<[1], offset: ?>>)
      params(%i : index)
      attributes {dep_modes = [#codir.access_mode<readwrite>]} {
    ^bb0(%dep: memref<?xi32, strided<[1], offset: ?>>, %i_arg: index):
      %v = memref.load %dep[%i_arg]
        : memref<?xi32, strided<[1], offset: ?>>
      memref.store %v, %dep[%i_arg]
        : memref<?xi32, strided<[1], offset: ?>>
      codir.yield
    }
    return
  }

  func.func @two_sliced_deps_from_same_root_materialize_positionally(%n: index,
                                                                     %off0: index,
                                                                     %off1: index,
                                                                     %i: index) {
    %root = sde.mu_alloc(%n) : memref<?xi32>
    %view0 = memref.subview %root[%off0] [%n] [1]
      : memref<?xi32> to memref<?xi32, strided<[1], offset: ?>>
    %view1 = memref.subview %root[%off1] [%n] [1]
      : memref<?xi32> to memref<?xi32, strided<[1], offset: ?>>

    codir.codelet deps(%view0, %view1
                       : memref<?xi32, strided<[1], offset: ?>>,
                         memref<?xi32, strided<[1], offset: ?>>)
      params(%i : index)
      attributes {dep_modes = [#codir.access_mode<read>,
                               #codir.access_mode<write>]} {
    ^bb0(%in: memref<?xi32, strided<[1], offset: ?>>,
         %out: memref<?xi32, strided<[1], offset: ?>>,
         %i_arg: index):
      %v = memref.load %in[%i_arg]
        : memref<?xi32, strided<[1], offset: ?>>
      memref.store %v, %out[%i_arg]
        : memref<?xi32, strided<[1], offset: ?>>
      codir.yield
    }
    return
  }
}

// CHECK-LABEL: func.func @dynamic_sliced_dep_remaps_edt_params
// CHECK: arts.db_acquire
// CHECK: arts.edt
// CHECK-SAME: params(%arg2, %arg1, %arg0 : index, index, index)
// CHECK: ^bb0(%[[DB:[a-zA-Z0-9_]+]]: {{.*}}, %[[I:[a-zA-Z0-9_]+]]: index, %[[OFF:[a-zA-Z0-9_]+]]: index, %[[N:[a-zA-Z0-9_]+]]: index):
// CHECK: %[[PAYLOAD:.*]] = arts.db_ref %[[DB]]
// CHECK: %[[VIEW:.*]] = memref.subview %[[PAYLOAD]][%[[OFF]]] [%[[N]]] [1]
// CHECK: memref.load %[[VIEW]][%[[I]]]
// CHECK: memref.store {{.*}}, %[[VIEW]][%[[I]]]

// CHECK-LABEL: func.func @two_sliced_deps_from_same_root_materialize_positionally
// CHECK: arts.db_acquire
// CHECK: arts.db_acquire
// CHECK: arts.edt
// CHECK-SAME: params(%arg3, %arg1, %arg0, %arg2 : index, index, index, index)
// CHECK: ^bb0(%[[DB0:[a-zA-Z0-9_]+]]: {{.*}}, %[[DB1:[a-zA-Z0-9_]+]]: {{.*}}, %[[I2:[a-zA-Z0-9_]+]]: index, %[[OFF0:[a-zA-Z0-9_]+]]: index, %[[N2:[a-zA-Z0-9_]+]]: index, %[[OFF1:[a-zA-Z0-9_]+]]: index):
// CHECK: %[[PAYLOAD0:.*]] = arts.db_ref %[[DB0]]
// CHECK: %[[VIEW0:.*]] = memref.subview %[[PAYLOAD0]][%[[OFF0]]] [%[[N2]]] [1]
// CHECK: %[[PAYLOAD1:.*]] = arts.db_ref %[[DB1]]
// CHECK: %[[VIEW1:.*]] = memref.subview %[[PAYLOAD1]][%[[OFF1]]] [%[[N2]]] [1]
// CHECK: %[[V2:.*]] = memref.load %[[VIEW0]][%[[I2]]]
// CHECK: memref.store %[[V2]], %[[VIEW1]][%[[I2]]]
