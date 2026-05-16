// RUN: not %carts-compile %s --pass-pipeline='builtin.module(verify-arts-objects-only)' 2>&1 | %FileCheck %s --check-prefix=ARTS

module {
  func.func @sde_survives_boundary() {
    // ARTS: SDE operation 'sde.control_token' remains after the CODIR-to-ARTS boundary
    %tok = sde.control_token : !sde.completion
    sde.su_barrier(%tok : !sde.completion)
    func.return
  }

  func.func @scf_parallel_survives_boundary() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // ARTS: scf.parallel remains after the CODIR-to-ARTS boundary
    scf.parallel (%i) = (%c0) to (%c1) step (%c1) {
      scf.reduce
    }
    func.return
  }

  func.func @host_fallback_scf_parallel_allowed() attributes {sde.keep_host_openmp} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.parallel (%i) = (%c0) to (%c1) step (%c1) {
      scf.reduce
    }
    func.return
  }
}
