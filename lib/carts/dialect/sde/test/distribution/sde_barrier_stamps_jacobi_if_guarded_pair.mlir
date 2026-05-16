// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from sde-planning --pipeline codir-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Jacobi benchmark loops lower as adjacent SDE stages in an outer timestep
// loop, with the second phase guarded by the timestep IV rather than separated
// by an explicit SDE barrier. SDE recognizes this as a full timestep pair,
// inserts an explicit stage-completion boundary, and marks the pair as a CPS
// candidate until tokenized stencil dataflow exists.

// CHECK-LABEL: // -----// IR Dump After BarrierElimination (barrier-elimination) //----- //
// CHECK: func.func @jacobi_if_guarded_pair
// CHECK: scf.for
// CHECK: sde.su_iterate
// CHECK-SAME: classification(<stencil>)
// CHECK: } {
// CHECK-SAME: asyncStrategy = #sde.async_strategy<advance_stage>
// CHECK-SAME: cps_candidate_group_id = 0 : i64
// CHECK-SAME: cps_candidate_requires_tokenized_dataflow
// CHECK-SAME: cps_candidate_stage_count = 2 : i64
// CHECK-SAME: cps_candidate_stage_index = 0 : i64
// CHECK-SAME: pattern = #sde.pattern<stencil_tiling_nd>
// CHECK-SAME: repetitionStructure = #sde.repetition_structure<full_timestep>
// CHECK: %[[DONE:[0-9]+]] = sde.control_token : !sde.completion
// CHECK: sde.su_barrier(%[[DONE]] : !sde.completion)
// CHECK-SAME: barrierReason = #sde.barrier_reason<timestep_stage_boundary>
// CHECK: scf.if
// CHECK: sde.su_iterate
// CHECK-SAME: classification(<stencil>)
// CHECK: } {
// CHECK-SAME: asyncStrategy = #sde.async_strategy<advance_stage>
// CHECK-SAME: cps_candidate_group_id = 0 : i64
// CHECK-SAME: cps_candidate_requires_tokenized_dataflow
// CHECK-SAME: cps_candidate_stage_count = 2 : i64
// CHECK-SAME: cps_candidate_stage_index = 1 : i64
// CHECK-SAME: pattern = #sde.pattern<stencil_tiling_nd>
// CHECK-SAME: repetitionStructure = #sde.repetition_structure<full_timestep>
// CHECK-LABEL: // -----// IR Dump After ConvertCodirToArts (convert-codir-to-arts) //----- //
// CHECK: func.func @jacobi_if_guarded_pair
// CHECK: depPattern = #arts.dep_pattern<stencil_tiling_nd>
// CHECK-SAME: planAsyncStrategy = #arts.plan_async_strategy<advance_edt>
// CHECK-SAME: planRepetitionStructure = #arts.plan_repetition_structure<full_timestep>
// CHECK: arts.barrier {barrierReason = #arts.barrier_reason<timestep_stage_boundary>}
// CHECK: scf.if
// CHECK: depPattern = #arts.dep_pattern<stencil_tiling_nd>
// CHECK-SAME: planAsyncStrategy = #arts.plan_async_strategy<advance_edt>
// CHECK-SAME: planRepetitionStructure = #arts.plan_repetition_structure<full_timestep>
// CHECK-NOT: sde.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @jacobi_if_guarded_pair(%A: memref<18x18xf64>, %B: memref<18x18xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c16 = arith.constant 16 : index
    %c20 = arith.constant 20 : index
    scf.for %t = %c0 to %c20 step %c2 {
      sde.cu_region <parallel> {
        sde.su_iterate (%c1) to (%c16) step (%c1) {
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
          sde.yield
        }
        sde.yield
      }
      %remaining = arith.subi %c16, %t : index
      %run_second = arith.cmpi sge, %remaining, %c0 : index
      scf.if %run_second {
        sde.cu_region <parallel> {
          sde.su_iterate (%c1) to (%c16) step (%c1) {
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
            sde.yield
          }
          sde.yield
        }
      }
    }
    return
  }
}
