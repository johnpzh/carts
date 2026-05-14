// RUN: not %carts-compile %s --arts-config %arts_config --start-from initial-cleanup --pipeline initial-cleanup 2>&1 | %FileCheck %s

// CHECK: sde.cps carry operand #0 cannot carry bare data/pointer type
// CHECK-SAME: use sde.mu_token, sde.mu_dep, or sde.control_token

module {
  func.func @raw_pointer_carry(%ptr: !llvm.ptr) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %result = sde.su_iterate (%c0) to (%c1) step (%c1) iter_args(%arg0 = %ptr : !llvm.ptr) -> (!llvm.ptr) {
    ^bb0(%i: index, %carry: !llvm.ptr):
      sde.yield %carry : !llvm.ptr
    } {asyncStrategy = #sde.async_strategy<cps_chain>,
       cps_group_id = 0 : i64,
       cps_stage_count = 1 : i64,
       cps_stage_index = 0 : i64,
       repetitionStructure = #sde.repetition_structure<full_timestep>}
    return
  }
}
