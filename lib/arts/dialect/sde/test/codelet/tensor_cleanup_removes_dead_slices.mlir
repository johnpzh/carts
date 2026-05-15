// RUN: %carts-compile %s --arts-config %arts_config --pipeline initial-cleanup --start-from initial-cleanup 2>&1 | %FileCheck %s

// Legacy tensor cleanup is being retired. Keep this migration test as a
// memref-native codelet parse/verify check for the same single-element update
// shape that tensor subset ops used to model.

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
