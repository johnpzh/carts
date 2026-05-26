// RUN: not %carts-compile %s --pass-pipeline='builtin.module(verify-codir)' 2>&1 | %FileCheck %s

module {
  func.func @codir_bad_block_arg_count(%dep: memref<4xf32>, %param: index) {
    codir.codelet deps(%dep : memref<4xf32>) params(%param : index) attributes {dep_modes = [#codir.access_mode<read>]} {
      codir.yield
    }
    return
  }

  func.func @codir_bad_block_arg_type(%dep: memref<4xf32>, %param: index) {
    codir.codelet deps(%dep : memref<4xf32>) params(%param : index) attributes {dep_modes = [#codir.access_mode<read>]} {
    ^bb0(%dep_arg: memref<8xf32>, %param_arg: i64):
      codir.yield
    }
    return
  }
}

// CHECK: expects 2 CODIR block argument(s) (1 dep + 1 param); got 0
// CHECK: dep block argument #0 type ('memref<8xf32>') does not match dep operand type ('memref<4xf32>')
// CHECK: param block argument #0 type ('i64') does not match param operand type ('index')
