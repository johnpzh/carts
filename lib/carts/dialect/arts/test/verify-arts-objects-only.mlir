// RUN: not %carts-compile %s --pass-pipeline='builtin.module(verify-arts-objects-only)' 2>&1 | %FileCheck %s --check-prefix=ARTS

module {
  func.func @sde_survives_boundary() {
    // ARTS: SDE operation 'sde.control_token' remains after the CODIR-to-ARTS boundary
    %tok = sde.control_token : !sde.completion
    sde.su_barrier(%tok : !sde.completion)
    func.return
  }
}
