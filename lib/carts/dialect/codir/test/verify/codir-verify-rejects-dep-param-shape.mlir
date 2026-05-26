// RUN: not %carts-compile %s --pass-pipeline='builtin.module(verify-codir)' 2>&1 | %FileCheck %s

module {
  func.func @codir_bad_dep_param(%n: index, %dep: memref<4xf32>) {
    codir.codelet deps(%n : index) params(%dep : memref<4xf32>) attributes {dep_modes = [#codir.access_mode<read>]} {
    ^bb0(%n_arg: index, %dep_arg: memref<4xf32>):
      codir.yield
    }
    return
  }
}

// CHECK: CODIR dependency #0 must be a memref value, got 'index'
// CHECK: CODIR parameter #0 must be an integer, index, or float scalar; got 'memref<4xf32>'
