// RUN: not %carts-compile %s --pass-pipeline='builtin.module(verify-codir)' 2>&1 | %FileCheck %s

module {
  func.func @codir_bad_yield(%dep: memref<4xf32>, %param: index) {
    codir.codelet deps(%dep : memref<4xf32>) params(%param : index) attributes {dep_modes = [#codir.access_mode<read>]} {
    ^bb0(%dep_arg: memref<4xf32>, %param_arg: index):
      codir.yield %dep_arg : memref<4xf32>
    }
    return
  }
}

// CHECK: CODIR yield operand #0 must be an integer, index, or float scalar; got 'memref<4xf32>'
