// RUN: not %carts-compile %s --pass-pipeline='builtin.module(verify-codir)' 2>&1 | %FileCheck %s

// Memref defined outside the codelet must not be referenced from the body
// unless it is surfaced through the codir.codelet deps ABI.

module {
  func.func @codir_implicit_memref_capture(%outside: memref<4xf32>) {
    codir.codelet {
      %c0 = arith.constant 0 : index
      %v = memref.load %outside[%c0] : memref<4xf32>
      codir.yield %v : f32
    }
    return
  }
}

// CHECK: error: 'memref.load' op using value defined outside the region
