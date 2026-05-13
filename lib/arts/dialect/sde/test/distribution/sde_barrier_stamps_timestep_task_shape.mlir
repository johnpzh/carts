// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from openmp-to-arts --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// SDE recognizes same-shape dependent elementwise stages as timestep waves
// before ConvertSdeToArts. Core only receives translated plan attrs.

// CHECK-LABEL: // -----// IR Dump After BarrierElimination (barrier-elimination) //----- //
// CHECK: func.func @timestep_pair
// CHECK: arts_sde.su_iterate
// CHECK: } {
// CHECK-SAME: asyncStrategy = #arts_sde.async_strategy<advance_edt>
// CHECK-SAME: repetitionStructure = #arts_sde.repetition_structure<full_timestep>
// CHECK: arts_sde.su_barrier
// CHECK-SAME: barrierReason = #arts_sde.barrier_reason<timestep_stage_boundary>
// CHECK: arts_sde.su_iterate
// CHECK: } {
// CHECK-SAME: asyncStrategy = #arts_sde.async_strategy<advance_edt>
// CHECK-SAME: repetitionStructure = #arts_sde.repetition_structure<full_timestep>
// CHECK-LABEL: // -----// IR Dump After ConvertSdeToArts (convert-sde-to-arts) //----- //
// CHECK: func.func @timestep_pair
// CHECK: arts.edt <parallel> <intranode> route(%{{.*}}) attributes {
// CHECK-SAME: no_verify = #arts.no_verify
// CHECK: depPattern = #arts.dep_pattern<uniform>
// CHECK-SAME: planAsyncStrategy = #arts.plan_async_strategy<advance_edt>
// CHECK-SAME: planRepetitionStructure = #arts.plan_repetition_structure<full_timestep>
// CHECK: arts.barrier
// CHECK-NOT: arts_sde.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @timestep_pair(%A: memref<64xf64>, %B: memref<64xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    arts_sde.cu_region <parallel> scope(<local>) {
      arts_sde.su_iterate (%c0) to (%c64) step (%c1) {
      ^bb0(%i: index):
        %v = memref.load %A[%i] : memref<64xf64>
        memref.store %v, %B[%i] : memref<64xf64>
        arts_sde.yield
      }
      arts_sde.su_barrier
      arts_sde.su_iterate (%c0) to (%c64) step (%c1) {
      ^bb0(%i: index):
        %v = memref.load %B[%i] : memref<64xf64>
        memref.store %v, %A[%i] : memref<64xf64>
        arts_sde.yield
      }
      arts_sde.yield
    }
    return
  }
}
