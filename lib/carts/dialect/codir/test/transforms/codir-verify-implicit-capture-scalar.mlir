// RUN: not %carts-compile %s --pass-pipeline='builtin.module(verify-codir)' 2>&1 | %FileCheck %s

// Scalar defined outside the codelet must not be referenced from the body
// unless it is surfaced through the codir.codelet params ABI.

module {
  func.func @codir_implicit_scalar_capture(%n: index) {
    codir.codelet {
      %c1 = arith.constant 1 : index
      %sum = arith.addi %n, %c1 : index
      codir.yield %sum : index
    }
    return
  }
}

// CHECK: error: 'arith.addi' op using value defined outside the region
