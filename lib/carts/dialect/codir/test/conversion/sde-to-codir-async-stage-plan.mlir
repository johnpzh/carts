// RUN: %carts-compile %s --pass-pipeline='builtin.module(convert-sde-to-codir,verify-codir)' \
// RUN:   | %FileCheck %s --implicit-check-not="advance_edt"

module {
  func.func @async_stage_plan(%A: memref<8xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    sde.cu_region <parallel> {
      sde.su_iterate (%c0) to (%c8) step (%c1) {
      ^bb0(%i: index):
        %v = arith.index_cast %i : index to i32
        memref.store %v, %A[%i] : memref<8xi32>
        sde.yield
      } {asyncStrategy = #sde.async_strategy<advance_stage>,
         repetitionStructure = #sde.repetition_structure<full_timestep>}
      sde.yield
    }
    return
  }
}

// CHECK-LABEL: func.func @async_stage_plan
// CHECK: codir.codelet
// CHECK-SAME: async_strategy = #codir.async_strategy<advance_stage>
