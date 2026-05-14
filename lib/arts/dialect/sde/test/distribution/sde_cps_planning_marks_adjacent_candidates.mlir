// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from openmp-to-arts --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Adjacent full-timestep stages do not get attrs-only CPS candidate state.
// SDE inserts an explicit completion token and token-consuming timestep
// barrier before it stamps the candidate group.

// CHECK-LABEL: // -----// IR Dump After CpsPlanning (cps-planning) //----- //
// CHECK: func.func @adjacent_timestep_pair
// CHECK: scf.for
// CHECK: arts_sde.cu_region
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
// CHECK: arts_sde.cu_region
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
  func.func @adjacent_timestep_pair(%A: memref<18x18xf64>, %B: memref<18x18xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    scf.for %t = %c0 to %c4 step %c1 {
      arts_sde.cu_region <parallel> scope(<local>) {
        arts_sde.su_iterate (%c1) to (%c16) step (%c1) {
        ^bb0(%i: index):
          scf.for %j = %c1 to %c16 step %c1 {
            %im1 = arith.subi %i, %c1 : index
            %ip1 = arith.addi %i, %c1 : index
            %jm1 = arith.subi %j, %c1 : index
            %jp1 = arith.addi %j, %c1 : index
            %n = memref.load %A[%im1, %j] : memref<18x18xf64>
            %s = memref.load %A[%ip1, %j] : memref<18x18xf64>
            %w = memref.load %A[%i, %jm1] : memref<18x18xf64>
            %e = memref.load %A[%i, %jp1] : memref<18x18xf64>
            %s0 = arith.addf %n, %s : f64
            %s1 = arith.addf %w, %e : f64
            %sum = arith.addf %s0, %s1 : f64
            memref.store %sum, %B[%i, %j] : memref<18x18xf64>
          }
          arts_sde.yield
        } {asyncStrategy = #arts_sde.async_strategy<advance_edt>,
           repetitionStructure = #arts_sde.repetition_structure<full_timestep>}
        arts_sde.yield
      }
      %always = arith.cmpi uge, %t, %c0 : index
      scf.if %always {
      }
      arts_sde.cu_region <parallel> scope(<local>) {
        arts_sde.su_iterate (%c1) to (%c16) step (%c1) {
        ^bb0(%i: index):
          scf.for %j = %c1 to %c16 step %c1 {
            %im1 = arith.subi %i, %c1 : index
            %ip1 = arith.addi %i, %c1 : index
            %jm1 = arith.subi %j, %c1 : index
            %jp1 = arith.addi %j, %c1 : index
            %n = memref.load %B[%im1, %j] : memref<18x18xf64>
            %s = memref.load %B[%ip1, %j] : memref<18x18xf64>
            %w = memref.load %B[%i, %jm1] : memref<18x18xf64>
            %e = memref.load %B[%i, %jp1] : memref<18x18xf64>
            %s0 = arith.addf %n, %s : f64
            %s1 = arith.addf %w, %e : f64
            %sum = arith.addf %s0, %s1 : f64
            memref.store %sum, %A[%i, %j] : memref<18x18xf64>
          }
          arts_sde.yield
        } {asyncStrategy = #arts_sde.async_strategy<advance_edt>,
           repetitionStructure = #arts_sde.repetition_structure<full_timestep>}
        arts_sde.yield
      }
    }
    return
  }
}
