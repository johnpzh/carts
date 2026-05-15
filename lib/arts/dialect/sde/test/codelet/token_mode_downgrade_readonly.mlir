// RUN: %carts-compile %s --arts-config %arts_config --pipeline initial-cleanup --start-from initial-cleanup 2>&1 | %FileCheck %s

// Round-trip test for codelet bodies with read-only and readwrite memref
// tokens.

module {
  // A read-only codelet uses a <read> token and has no results.
  // CHECK-LABEL: func.func @readonly_codelet
  // CHECK-SAME: %[[M:.*]]: memref<8xi32>
  func.func @readonly_codelet(%m: memref<8xi32>) {
    // CHECK: sde.mu_token <read> %[[M]]
    %token = sde.mu_token <read> %m
      : memref<8xi32> -> !sde.token<memref<8xi32>>

    // CHECK: sde.cu_codelet
    sde.cu_codelet (%token : !sde.token<memref<8xi32>>) {
    ^bb0(%arg: memref<8xi32>):
      %c0 = arith.constant 0 : index
      %val = memref.load %arg[%c0] : memref<8xi32>
      %buf = memref.alloca() : memref<1xi32>
      memref.store %val, %buf[%c0] : memref<1xi32>
      sde.yield
    }

    func.return
  }

  // A readwrite codelet keeps readwrite since it writes in-place.
  // CHECK-LABEL: func.func @readwrite_codelet
  // CHECK-SAME: %[[M2:.*]]: memref<8xi32>
  func.func @readwrite_codelet(%m: memref<8xi32>) {
    // CHECK: sde.mu_token <readwrite> %[[M2]]
    %token = sde.mu_token <readwrite> %m
      : memref<8xi32> -> !sde.token<memref<8xi32>>

    // CHECK: sde.cu_codelet
    sde.cu_codelet (%token : !sde.token<memref<8xi32>>) {
    ^bb0(%arg: memref<8xi32>):
      %c0 = arith.constant 0 : index
      %val = memref.load %arg[%c0] : memref<8xi32>
      %c42 = arith.constant 42 : i32
      %sum = arith.addi %val, %c42 : i32
      memref.store %sum, %arg[%c0] : memref<8xi32>
      sde.yield
    }

    func.return
  }
}
