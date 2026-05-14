// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from openmp-to-arts --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// SDE stamps runtime-neutral task-shape plan attrs. ConvertSdeToArts is the
// boundary that translates them into Core ARTS plan attrs.

// CHECK-LABEL: // -----// IR Dump After ConvertSdeToArts (convert-sde-to-arts) //----- //
// CHECK: func.func @main
// CHECK: arts.epoch attributes {
// CHECK-SAME: planAsyncStrategy = #arts.plan_async_strategy<advance_edt>
// CHECK-SAME: planRepetitionStructure = #arts.plan_repetition_structure<full_timestep>
// CHECK: arts.edt <task>
// CHECK-SAME: planAsyncStrategy = #arts.plan_async_strategy<advance_edt>
// CHECK-SAME: planRepetitionStructure = #arts.plan_repetition_structure<full_timestep>
// CHECK-NOT: arts_sde.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<8xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    arts_sde.cu_region <parallel> scope(<local>) {
      arts_sde.su_iterate (%c0) to (%c8) step (%c1) {
      ^bb0(%i: index):
        %v = arith.index_cast %i : index to i32
        memref.store %v, %A[%i] : memref<8xi32>
        arts_sde.yield
      } {asyncStrategy = #arts_sde.async_strategy<advance_edt>, repetitionStructure = #arts_sde.repetition_structure<full_timestep>}
      arts_sde.yield
    }
    return
  }
}
