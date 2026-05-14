// RUN: %carts-compile %s --arts-config %arts_config --pipeline initial-cleanup --start-from initial-cleanup 2>&1 | %FileCheck %s

// Phase 0 verification: rank-0 tensor (scalar) support through
// mu_data / mu_token / cu_codelet. This is a prerequisite for the
// tensor-first pipeline — cu_region <single> bodies operate on scalars
// like `sum += 1000` which become tensor<i32> after raising.

module {
  // Rank-0 mu_data roundtrip.
  // CHECK-LABEL: func.func @mu_data_rank0
  func.func @mu_data_rank0() -> tensor<i32> {
    // CHECK: %[[D:.*]] = sde.mu_data shared : tensor<i32>
    %d = sde.mu_data shared : tensor<i32>
    // CHECK: return %[[D]] : tensor<i32>
    func.return %d : tensor<i32>
  }

  // Rank-0 whole-tensor token + codelet roundtrip: readwrite scalar.
  // CHECK-LABEL: func.func @codelet_rank0_readwrite
  // CHECK-SAME: %[[T:.*]]: tensor<i32>
  func.func @codelet_rank0_readwrite(%t: tensor<i32>) -> tensor<i32> {
    // CHECK: %[[TOK:.*]] = sde.mu_token <readwrite> %[[T]] : tensor<i32> -> !sde.token<tensor<i32>>
    %token = sde.mu_token <readwrite> %t
      : tensor<i32> -> !sde.token<tensor<i32>>

    // CHECK: %[[R:.*]] = sde.cu_codelet(%[[TOK]] : !sde.token<tensor<i32>>) -> (tensor<i32>)
    %r = sde.cu_codelet (%token : !sde.token<tensor<i32>>)
        -> (tensor<i32>) {
    // CHECK: ^bb0(%[[ARG:.*]]: tensor<i32>)
    ^bb0(%arg: tensor<i32>):
      %val = tensor.extract %arg[] : tensor<i32>
      %c1000 = arith.constant 1000 : i32
      %added = arith.addi %val, %c1000 : i32
      %result = tensor.insert %added into %arg[] : tensor<i32>
      sde.yield %result : tensor<i32>
    }

    // CHECK: return %[[R]] : tensor<i32>
    func.return %r : tensor<i32>
  }

  // Rank-0 read-only codelet: no result.
  // CHECK-LABEL: func.func @codelet_rank0_read_only
  func.func @codelet_rank0_read_only(%t: tensor<i32>) {
    // CHECK: sde.mu_token <read>
    %token = sde.mu_token <read> %t
      : tensor<i32> -> !sde.token<tensor<i32>>

    // CHECK: sde.cu_codelet
    sde.cu_codelet (%token : !sde.token<tensor<i32>>) {
    ^bb0(%arg: tensor<i32>):
      %val = tensor.extract %arg[] : tensor<i32>
      %buf = memref.alloca() : memref<1xi32>
      %c0 = arith.constant 0 : index
      memref.store %val, %buf[%c0] : memref<1xi32>
      sde.yield
    }
    func.return
  }
}
