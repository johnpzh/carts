// RUN: %carts-compile %s --arts-config %arts_config --pipeline initial-cleanup --start-from initial-cleanup 2>&1 | %FileCheck %s

// Positive round-trip tests for the memref-native SDE codelet surface:
//   - !sde.token<memref<...>> types
//   - sde.mu_data (declarative SDE shared-data handle)
//   - sde.mu_token (MU storage access token producer)
//   - sde.cu_codelet (dataflow-isolated compute unit)
//
// The initial-cleanup stage only parses, verifies, and prints these ops.

module {
  // CHECK-LABEL: func.func @codelet_roundtrip
  // CHECK-SAME: %[[M:.*]]: memref<8xi32>
  func.func @codelet_roundtrip(%m: memref<8xi32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index

    // CHECK: %[[TOK:.*]] = sde.mu_token <readwrite> %[[M]][%{{[^ ]+}}] size[%{{[^ ]+}}] : memref<8xi32> -> !sde.token<memref<4xi32>>
    %token = sde.mu_token <readwrite> %m [%c0] size [%c4]
      : memref<8xi32> -> !sde.token<memref<4xi32>>

    // CHECK: sde.cu_codelet(%[[TOK]] : !sde.token<memref<4xi32>>)
    sde.cu_codelet (%token : !sde.token<memref<4xi32>>) {
    // CHECK: ^bb0(%[[ARG:.*]]: memref<4xi32>)
    ^bb0(%arg: memref<4xi32>):
      %local_c0 = arith.constant 0 : index
      %v = memref.load %arg[%local_c0] : memref<4xi32>
      memref.store %v, %arg[%local_c0] : memref<4xi32>
      // CHECK: sde.yield
      sde.yield
    }

    func.return
  }

  // CHECK-LABEL: func.func @mu_data_roundtrip
  func.func @mu_data_roundtrip() -> memref<8xi32> {
    // CHECK: %[[D:.*]] = sde.mu_data shared : memref<8xi32>
    %d = sde.mu_data shared : memref<8xi32>
    // CHECK: return %[[D]] : memref<8xi32>
    func.return %d : memref<8xi32>
  }

  // Whole-storage token (no offsets/sizes) round-trips.
  // CHECK-LABEL: func.func @whole_memref_token_with_dataflow_body
  func.func @whole_memref_token_with_dataflow_body(%m: memref<8xi32>) {
    %c0 = arith.constant 0 : index
    // CHECK: %[[TOK:.*]] = sde.mu_token <readwrite> %{{[^ ]+}} : memref<8xi32> -> !sde.token<memref<8xi32>>
    %token = sde.mu_token <readwrite> %m
      : memref<8xi32> -> !sde.token<memref<8xi32>>

    // CHECK: sde.cu_codelet(%[[TOK]] : !sde.token<memref<8xi32>>)
    sde.cu_codelet (%token : !sde.token<memref<8xi32>>) {
    ^bb0(%arg: memref<8xi32>):
      %local_c0 = arith.constant 0 : index
      %v = memref.load %arg[%local_c0] : memref<8xi32>
      memref.store %v, %arg[%local_c0] : memref<8xi32>
      sde.yield
    }
    func.return
  }

  // Memref tokens are direct mutable views. A writable memref token does not
  // require a destination-passing codelet result.
  // CHECK-LABEL: func.func @whole_memref_token
  // CHECK-SAME: %[[M:.*]]: memref<8xi32>
  func.func @whole_memref_token(%m: memref<8xi32>) {
    // CHECK: %[[TOK:.*]] = sde.mu_token <readwrite> %[[M]] : memref<8xi32> -> !sde.token<memref<8xi32>>
    %token = sde.mu_token <readwrite> %m
      : memref<8xi32> -> !sde.token<memref<8xi32>>

    // CHECK: sde.cu_codelet(%[[TOK]] : !sde.token<memref<8xi32>>)
    sde.cu_codelet (%token : !sde.token<memref<8xi32>>) {
    // CHECK: ^bb0(%[[ARG:.*]]: memref<8xi32>)
    ^bb0(%arg: memref<8xi32>):
      %c0 = arith.constant 0 : index
      %v = memref.load %arg[%c0] : memref<8xi32>
      memref.store %v, %arg[%c0] : memref<8xi32>
      // CHECK: sde.yield
      sde.yield
    }
    func.return
  }
}
