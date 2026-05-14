// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from openmp-to-arts --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// A standalone out-of-place stencil stays stencil_tiling_nd. A same-shape
// copy/stencil pair with alternating read/write roots is the SDE-owned Jacobi
// timestep pattern.

// CHECK-LABEL: // -----// IR Dump After BarrierElimination (barrier-elimination) //----- //
// CHECK: func.func @jacobi_pair
// CHECK: arts_sde.su_iterate
// CHECK: } {
// CHECK-SAME: depFamily = #arts_sde.dep_family<uniform>
// CHECK-SAME: repetitionStructure = #arts_sde.repetition_structure<full_timestep>
// CHECK: arts_sde.su_barrier
// CHECK-SAME: barrierReason = #arts_sde.barrier_reason<timestep_stage_boundary>
// CHECK: arts_sde.su_iterate
// CHECK-SAME: classification(<stencil>)
// CHECK: } {
// CHECK-SAME: depFamily = #arts_sde.dep_family<jacobi_alternating_buffers>
// CHECK-SAME: repetitionStructure = #arts_sde.repetition_structure<full_timestep>
// CHECK-LABEL: // -----// IR Dump After ConvertSdeToArts (convert-sde-to-arts) //----- //
// CHECK: func.func @jacobi_pair
// CHECK: arts.epoch attributes {
// CHECK-SAME: planRepetitionStructure = #arts.plan_repetition_structure<full_timestep>
// CHECK: arts.edt <task>
// CHECK-SAME: depPattern = #arts.dep_pattern<uniform>
// CHECK-SAME: no_verify = #arts.no_verify
// CHECK-SAME: planRepetitionStructure = #arts.plan_repetition_structure<full_timestep>
// CHECK: arts.barrier
// CHECK-SAME: barrierReason = #arts.barrier_reason<timestep_stage_boundary>
// CHECK: arts.edt <task>
// CHECK-SAME: depPattern = #arts.dep_pattern<jacobi_alternating_buffers>
// CHECK-SAME: planRepetitionStructure = #arts.plan_repetition_structure<full_timestep>
// CHECK-NOT: arts_sde.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @jacobi_pair(%A: memref<18x18xf64>, %B: memref<18x18xf64>) {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    arts_sde.cu_region <parallel> scope(<local>) {
      arts_sde.su_iterate (%c1) to (%c16) step (%c1) {
      ^bb0(%i: index):
        scf.for %j = %c1 to %c16 step %c1 {
          %v = memref.load %B[%i, %j] : memref<18x18xf64>
          memref.store %v, %A[%i, %j] : memref<18x18xf64>
        }
        arts_sde.yield
      }
      arts_sde.su_barrier
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
      }
      arts_sde.yield
    }
    return
  }
}
