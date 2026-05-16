// RUN: %carts-compile %s --arts-config %arts_config --pipeline initial-cleanup --start-from initial-cleanup 2>&1 | %FileCheck %s

// Memref-native codelets should parse and verify single-element update shapes
// without tensor subset cleanup.

module {
  // CHECK-LABEL: func.func @codelet_with_single_element_update
  // CHECK-SAME: %[[M:.*]]: memref<16xf64>
  func.func @codelet_with_single_element_update(%m: memref<16xf64>) {
    // CHECK: sde.mu_token <readwrite>
    %token = sde.mu_token <readwrite> %m
      : memref<16xf64> -> !sde.token<memref<16xf64>>

    // CHECK: sde.cu_codelet
    sde.cu_codelet (%token : !sde.token<memref<16xf64>>) {
    ^bb0(%arg: memref<16xf64>):
      %c0 = arith.constant 0 : index
      %val = memref.load %arg[%c0] : memref<16xf64>
      %c1 = arith.constant 1.0 : f64
      %sum = arith.addf %val, %c1 : f64
      memref.store %sum, %arg[%c0] : memref<16xf64>
      sde.yield
    }

    func.return
  }
}
