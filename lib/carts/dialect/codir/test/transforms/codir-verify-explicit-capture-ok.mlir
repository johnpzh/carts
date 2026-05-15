// RUN: %carts-compile %s --pass-pipeline='builtin.module(verify-codir)' | %FileCheck %s

// Positive: every body operand is either a body-local SSA value or a
// dep/param block argument. verify-codir must accept this codelet and the
// IR must round-trip unchanged.

module {
  func.func @codir_explicit_capture_ok(%dep: memref<4xf32>, %n: index) {
    codir.codelet deps(%dep : memref<4xf32>) params(%n : index)
        attributes {dep_modes = [#codir.access_mode<readwrite>]} {
    ^bb0(%dep_arg: memref<4xf32>, %n_arg: index):
      %c0 = arith.constant 0 : index
      %v = memref.load %dep_arg[%c0] : memref<4xf32>
      %as_idx = arith.index_cast %n_arg : index to i64
      %as_f = arith.sitofp %as_idx : i64 to f32
      %sum = arith.addf %v, %as_f : f32
      memref.store %sum, %dep_arg[%c0] : memref<4xf32>
      codir.yield %sum : f32
    }
    return
  }
}

// CHECK-LABEL: func.func @codir_explicit_capture_ok
// CHECK: codir.codelet deps(%{{.*}} : memref<4xf32>) params(%{{.*}} : index)
// CHECK: ^bb0(%{{.*}}: memref<4xf32>, %{{.*}}: index):
// CHECK: codir.yield
