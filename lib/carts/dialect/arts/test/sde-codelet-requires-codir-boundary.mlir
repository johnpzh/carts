// RUN: not %carts-compile %s --arts-config %arts_config --start-from codir-to-arts --pipeline codir-to-arts 2>&1 | %FileCheck %s

// CODIR-to-ARTS rejects residual SDE codelet ops at the boundary.

module {
  func.func @direct_sde_codelet_rejected() {
    %d = sde.mu_data shared : memref<8xi32>
    %t = sde.mu_token <readwrite> %d
      : memref<8xi32> -> !sde.token<memref<8xi32>>

    // CHECK: sde.mu_token
    // CHECK: survived CODIR-to-ARTS materialization
    // CHECK: sde.cu_codelet
    sde.cu_codelet (%t : !sde.token<memref<8xi32>>) {
    ^bb0(%arg0: memref<8xi32>):
      sde.yield
    }
    func.return
  }
}
