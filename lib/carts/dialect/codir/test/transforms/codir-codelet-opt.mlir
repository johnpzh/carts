// RUN: %carts-compile %s --pass-pipeline='builtin.module(codir-codelet-opt,verify-codir)' | %FileCheck %s

module {
  func.func @codelet_opt_removes_dead_token_local_rematerialization(%dep: memref<8xi32>, %i: index) {
    codir.codelet deps(%dep : memref<8xi32>) params(%i : index)
      attributes {dep_modes = [#codir.access_mode<readwrite>]} {
    ^bb0(%dep_arg: memref<8xi32>, %i_arg: index):
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %dead = arith.addi %c2, %c2 : index
      %dead_view = memref.subview %dep_arg[2] [2] [1]
        : memref<8xi32> to memref<2xi32, strided<[1], offset: 2>>
      %v = memref.load %dep_arg[%c0] : memref<8xi32>
      memref.store %v, %dep_arg[%c0] : memref<8xi32>
      codir.yield
    }
    return
  }
}

// CHECK-LABEL: func.func @codelet_opt_removes_dead_token_local_rematerialization
// CHECK: codir.codelet
// CHECK-NOT: arith.addi
// CHECK-NOT: memref.subview
// CHECK: memref.load
// CHECK: memref.store
// CHECK: codir.yield
