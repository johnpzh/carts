// RUN: %carts-compile %s --arts-config %arts_config --pipeline initial-cleanup --start-from initial-cleanup 2>&1 | %FileCheck %s

// Rank-0 memref support through mu_data / mu_token / cu_codelet.

module {
  // Rank-0 mu_data roundtrip.
  // CHECK-LABEL: func.func @mu_data_rank0
  func.func @mu_data_rank0() -> memref<i32> {
    // CHECK: %[[D:.*]] = sde.mu_data shared : memref<i32>
    %d = sde.mu_data shared : memref<i32>
    // CHECK: return %[[D]] : memref<i32>
    func.return %d : memref<i32>
  }

  // Rank-0 whole-storage token + codelet roundtrip: readwrite scalar.
  // CHECK-LABEL: func.func @codelet_rank0_readwrite
  // CHECK-SAME: %[[M:.*]]: memref<i32>
  func.func @codelet_rank0_readwrite(%m: memref<i32>) {
    // CHECK: %[[TOK:.*]] = sde.mu_token <readwrite> %[[M]] : memref<i32> -> !sde.token<memref<i32>>
    %token = sde.mu_token <readwrite> %m
      : memref<i32> -> !sde.token<memref<i32>>

    // CHECK: sde.cu_codelet(%[[TOK]] : !sde.token<memref<i32>>)
    sde.cu_codelet (%token : !sde.token<memref<i32>>) {
    // CHECK: ^bb0(%[[ARG:.*]]: memref<i32>)
    ^bb0(%arg: memref<i32>):
      %val = memref.load %arg[] : memref<i32>
      %c1000 = arith.constant 1000 : i32
      %added = arith.addi %val, %c1000 : i32
      memref.store %added, %arg[] : memref<i32>
      sde.yield
    }

    func.return
  }

  // Rank-0 read-only codelet: no result.
  // CHECK-LABEL: func.func @codelet_rank0_read_only
  func.func @codelet_rank0_read_only(%m: memref<i32>) {
    // CHECK: sde.mu_token <read>
    %token = sde.mu_token <read> %m
      : memref<i32> -> !sde.token<memref<i32>>

    // CHECK: sde.cu_codelet
    sde.cu_codelet (%token : !sde.token<memref<i32>>) {
    ^bb0(%arg: memref<i32>):
      %val = memref.load %arg[] : memref<i32>
      %buf = memref.alloca() : memref<i32>
      memref.store %val, %buf[] : memref<i32>
      sde.yield
    }
    func.return
  }
}
