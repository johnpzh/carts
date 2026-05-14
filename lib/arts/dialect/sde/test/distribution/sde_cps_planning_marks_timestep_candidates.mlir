// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from openmp-to-arts --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// SDE barrier analysis identifies the repeated timestep pair. SDE CPS planning
// groups it as a candidate, emits a stage-completion token for the barrier
// edge, and leaves asyncStrategy on advance_edt until every carry is explicit.

// CHECK-LABEL: // -----// IR Dump After CpsPlanning (cps-planning) //----- //
// CHECK: func.func @timestep_pair
// CHECK: arts_sde.su_iterate
// CHECK: } {
// CHECK-SAME: asyncStrategy = #arts_sde.async_strategy<advance_edt>
// CHECK-SAME: cps_candidate_group_id = 0 : i64
// CHECK-SAME: cps_candidate_requires_tokenized_dataflow
// CHECK-SAME: cps_candidate_stage_count = 2 : i64
// CHECK-SAME: cps_candidate_stage_index = 0 : i64
// CHECK-SAME: repetitionStructure = #arts_sde.repetition_structure<full_timestep>
// CHECK: %[[DONE:[0-9]+]] = arts_sde.control_token : !arts_sde.completion
// CHECK: arts_sde.su_barrier(%[[DONE]] : !arts_sde.completion)
// CHECK-SAME: barrierReason = #arts_sde.barrier_reason<timestep_stage_boundary>
// CHECK: arts_sde.su_iterate
// CHECK: } {
// CHECK-SAME: asyncStrategy = #arts_sde.async_strategy<advance_edt>
// CHECK-SAME: cps_candidate_group_id = 0 : i64
// CHECK-SAME: cps_candidate_requires_tokenized_dataflow
// CHECK-SAME: cps_candidate_stage_count = 2 : i64
// CHECK-SAME: cps_candidate_stage_index = 1 : i64
// CHECK-SAME: repetitionStructure = #arts_sde.repetition_structure<full_timestep>
// CHECK-LABEL: // -----// IR Dump After ConvertSdeToArts (convert-sde-to-arts) //----- //
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
