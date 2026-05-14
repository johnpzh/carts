// RUN: %carts-compile %s --arts-config %arts_config --pipeline initial-cleanup --start-from initial-cleanup 2>&1 | %FileCheck %s

// Verify that the tensor-cleanup pass can round-trip a codelet containing
// tensor subset ops. The initial-cleanup stage is a no-op pass-through;
// this test validates that the IR with tensor artifacts parses and verifies.

module {
  // CHECK-LABEL: func.func @codelet_with_slices
  // CHECK-SAME: %[[T:.*]]: tensor<16xf64>
  func.func @codelet_with_slices(%t: tensor<16xf64>) -> tensor<16xf64> {
    // CHECK: sde.mu_token <readwrite>
    %token = sde.mu_token <readwrite> %t
      : tensor<16xf64> -> !sde.token<tensor<16xf64>>

    // CHECK: sde.cu_codelet
    %r = sde.cu_codelet (%token : !sde.token<tensor<16xf64>>)
        -> (tensor<16xf64>) {
    ^bb0(%arg: tensor<16xf64>):
      %c0 = arith.constant 0 : index
      %val = tensor.extract %arg[%c0] : tensor<16xf64>
      %c1 = arith.constant 1.0 : f64
      %sum = arith.addf %val, %c1 : f64
      %out = tensor.insert %sum into %arg[%c0] : tensor<16xf64>
      sde.yield %out : tensor<16xf64>
    }

    func.return %r : tensor<16xf64>
  }
}
