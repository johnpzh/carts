// RUN: not %carts-compile %s --pass-pipeline='builtin.module(verify-codir)' 2>&1 | %FileCheck %s

module {
  func.func @codir_missing_dep_modes(%dep: memref<4xf32>) {
    codir.codelet deps(%dep : memref<4xf32>) {
    ^bb0(%dep_arg: memref<4xf32>):
      codir.yield
    }
    return
  }

  func.func @codir_bad_dep_mode_count(%dep0: memref<4xf32>, %dep1: memref<4xf32>) {
    codir.codelet deps(%dep0, %dep1 : memref<4xf32>, memref<4xf32>) attributes {dep_modes = [#codir.access_mode<read>]} {
    ^bb0(%dep0_arg: memref<4xf32>, %dep1_arg: memref<4xf32>):
      codir.yield
    }
    return
  }
}

// CHECK: expects dep_modes attribute when dependency operands are present
// CHECK: expects dep_modes entry count (1) to match dependency operand count (2)
