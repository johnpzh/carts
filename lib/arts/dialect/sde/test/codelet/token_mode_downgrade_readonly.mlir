// RUN: %carts-compile %s --arts-config %arts_config --pipeline initial-cleanup --start-from initial-cleanup 2>&1 | %FileCheck %s

// Round-trip test for codelet bodies with read-only and readwrite tokens.
// The token-mode-refinement pass (run inside openmp-to-arts) would downgrade
// readwrite tokens to read when the codelet body has no writes.

module {
  // A read-only codelet — uses <read> token, no results.
  // CHECK-LABEL: func.func @readonly_codelet
  // CHECK-SAME: %[[T:.*]]: tensor<8xi32>
  func.func @readonly_codelet(%t: tensor<8xi32>) {
    // CHECK: sde.mu_token <read> %[[T]]
    %token = sde.mu_token <read> %t
      : tensor<8xi32> -> !sde.token<tensor<8xi32>>

    // CHECK: sde.cu_codelet
    sde.cu_codelet (%token : !sde.token<tensor<8xi32>>) {
    ^bb0(%arg: tensor<8xi32>):
      %c0 = arith.constant 0 : index
      %val = tensor.extract %arg[%c0] : tensor<8xi32>
      %buf = memref.alloca() : memref<1xi32>
      memref.store %val, %buf[%c0] : memref<1xi32>
      sde.yield
    }

    func.return
  }

  // A readwrite codelet — should keep readwrite since it writes.
  // CHECK-LABEL: func.func @readwrite_codelet
  // CHECK-SAME: %[[T2:.*]]: tensor<8xi32>
  func.func @readwrite_codelet(%t: tensor<8xi32>) -> tensor<8xi32> {
    // CHECK: sde.mu_token <readwrite> %[[T2]]
    %token = sde.mu_token <readwrite> %t
      : tensor<8xi32> -> !sde.token<tensor<8xi32>>

    // CHECK: sde.cu_codelet
    %r = sde.cu_codelet (%token : !sde.token<tensor<8xi32>>)
        -> (tensor<8xi32>) {
    ^bb0(%arg: tensor<8xi32>):
      %c0 = arith.constant 0 : index
      %val = tensor.extract %arg[%c0] : tensor<8xi32>
      %c42 = arith.constant 42 : i32
      %sum = arith.addi %val, %c42 : i32
      %out = tensor.insert %sum into %arg[%c0] : tensor<8xi32>
      sde.yield %out : tensor<8xi32>
    }

    func.return %r : tensor<8xi32>
  }
}
