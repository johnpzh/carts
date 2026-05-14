// RUN: %carts-compile %s --arts-config %arts_config --pipeline initial-cleanup --start-from initial-cleanup 2>&1 | %FileCheck %s

// Positive round-trip tests for the RFC step-1 SDE ops:
//   - !sde.token<tensor<...>> type
//   - sde.mu_data (declarative SDE shared-data handle)
//   - sde.mu_token (tensor-path access token producer)
//   - sde.cu_codelet (dataflow-isolated compute unit)
//
// We run the benign `initial-cleanup` stage (LowerAffine + CSE +
// PolygeistCanonicalizeFor, all func-nested) so the only work performed on the
// module is parse + verify + print. Values that flow into `func.return` keep
// the ops alive across CSE/DCE.

module {
  // CHECK-LABEL: func.func @codelet_roundtrip
  // CHECK-SAME: %[[T:.*]]: tensor<8xi32>
  func.func @codelet_roundtrip(%t: tensor<8xi32>) -> tensor<8xi32> {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index

    // CHECK: %[[TOK:.*]] = sde.mu_token <readwrite> %[[T]][%{{[^ ]+}}] size[%{{[^ ]+}}] : tensor<8xi32> -> !sde.token<tensor<4xi32>>
    %token = sde.mu_token <readwrite> %t [%c0] size [%c4]
      : tensor<8xi32> -> !sde.token<tensor<4xi32>>

    // CHECK: %[[R:.*]] = sde.cu_codelet(%[[TOK]] : !sde.token<tensor<4xi32>>) -> (tensor<8xi32>)
    %r = sde.cu_codelet (%token : !sde.token<tensor<4xi32>>)
        -> (tensor<8xi32>) {
    // CHECK: ^bb0(%[[ARG:.*]]: tensor<4xi32>)
    ^bb0(%arg: tensor<4xi32>):
      // CHECK: sde.yield %[[ARG]] : tensor<4xi32>
      sde.yield %arg : tensor<4xi32>
    }

    // CHECK: return %[[R]] : tensor<8xi32>
    func.return %r : tensor<8xi32>
  }

  // CHECK-LABEL: func.func @mu_data_roundtrip
  func.func @mu_data_roundtrip() -> tensor<8xi32> {
    // CHECK: %[[D:.*]] = sde.mu_data shared : tensor<8xi32>
    %d = sde.mu_data shared : tensor<8xi32>
    // CHECK: return %[[D]] : tensor<8xi32>
    func.return %d : tensor<8xi32>
  }

  // Whole-tensor token (no offsets/sizes) round-trips; we yield the
  // untouched input so the codelet survives CSE/DCE.
  // CHECK-LABEL: func.func @whole_tensor_token
  func.func @whole_tensor_token(%t: tensor<8xi32>) -> tensor<8xi32> {
    // CHECK: %[[TOK:.*]] = sde.mu_token <readwrite> %{{[^ ]+}} : tensor<8xi32> -> !sde.token<tensor<8xi32>>
    %token = sde.mu_token <readwrite> %t
      : tensor<8xi32> -> !sde.token<tensor<8xi32>>

    // CHECK: %[[R:.*]] = sde.cu_codelet(%[[TOK]] : !sde.token<tensor<8xi32>>) -> (tensor<8xi32>)
    %r = sde.cu_codelet (%token : !sde.token<tensor<8xi32>>)
        -> (tensor<8xi32>) {
    ^bb0(%arg: tensor<8xi32>):
      sde.yield %arg : tensor<8xi32>
    }
    func.return %r : tensor<8xi32>
  }
}
