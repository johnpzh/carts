// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from sde-planning --pipeline codir-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// SDE recognizes same-shape dependent elementwise stages as timestep waves.
// Barrier analysis keeps them on advance_stage until the SDE CPS dataflow rewrite
// can make token/control carries explicit.

// CHECK-LABEL: // -----// IR Dump After BarrierElimination (barrier-elimination) //----- //
// CHECK: func.func @timestep_pair
// CHECK: sde.su_iterate
// CHECK: } {
// CHECK-SAME: asyncStrategy = #sde.async_strategy<advance_stage>
// CHECK-SAME: repetitionStructure = #sde.repetition_structure<full_timestep>
// CHECK: sde.su_barrier
// CHECK-SAME: barrierReason = #sde.barrier_reason<timestep_stage_boundary>
// CHECK: sde.su_iterate
// CHECK: } {
// CHECK-SAME: asyncStrategy = #sde.async_strategy<advance_stage>
// CHECK-SAME: repetitionStructure = #sde.repetition_structure<full_timestep>
// CHECK-LABEL: // -----// IR Dump After ConvertCodirToArts (convert-codir-to-arts) //----- //
// CHECK: func.func @timestep_pair
// CHECK: arts.edt <task>
// CHECK-SAME: planAsyncStrategy = #arts.plan_async_strategy<advance_edt>
// CHECK-SAME: planRepetitionStructure = #arts.plan_repetition_structure<full_timestep>
// CHECK: arts.edt <task>
// CHECK-SAME: depPattern = #arts.dep_pattern<uniform>
// CHECK-SAME: planAsyncStrategy = #arts.plan_async_strategy<advance_edt>
// CHECK-SAME: planRepetitionStructure = #arts.plan_repetition_structure<full_timestep>
// CHECK: arts.barrier
// CHECK-NOT: sde.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @timestep_pair(%A: memref<64xf64>, %B: memref<64xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    sde.cu_region <parallel> {
      sde.su_iterate (%c0) to (%c64) step (%c1) {
      ^bb0(%i: index):
        %v = memref.load %A[%i] : memref<64xf64>
        memref.store %v, %B[%i] : memref<64xf64>
        sde.yield
      }
      sde.su_barrier
      sde.su_iterate (%c0) to (%c64) step (%c1) {
      ^bb0(%i: index):
        %v = memref.load %B[%i] : memref<64xf64>
        memref.store %v, %A[%i] : memref<64xf64>
        sde.yield
      }
      sde.yield
    }
    return
  }
}
