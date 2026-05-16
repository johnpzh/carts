// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from sde-planning --pipeline codir-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// A standalone out-of-place stencil stays stencil_tiling_nd. A same-shape
// copy/stencil pair with alternating read/write roots is an SDE Jacobi
// timestep candidate. It stays on advance_edt until SDE rewrites the pair into
// explicit CPS dataflow tokens.

// CHECK-LABEL: // -----// IR Dump After BarrierElimination (barrier-elimination) //----- //
// CHECK: func.func @jacobi_pair
// CHECK: sde.su_iterate
// CHECK: } {
// CHECK-SAME: asyncStrategy = #sde.async_strategy<advance_edt>
// CHECK-SAME: pattern = #sde.pattern<uniform>
// CHECK-SAME: repetitionStructure = #sde.repetition_structure<full_timestep>
// CHECK: sde.su_barrier
// CHECK-SAME: barrierReason = #sde.barrier_reason<timestep_stage_boundary>
// CHECK: sde.su_iterate
// CHECK-SAME: classification(<stencil>)
// CHECK: } {
// CHECK-SAME: asyncStrategy = #sde.async_strategy<advance_edt>
// CHECK-SAME: pattern = #sde.pattern<jacobi_alternating_buffers>
// CHECK-SAME: repetitionStructure = #sde.repetition_structure<full_timestep>
// CHECK-LABEL: // -----// IR Dump After ConvertCodirToArts (convert-codir-to-arts) //----- //
// CHECK: func.func @jacobi_pair
// CHECK: arts.edt <task>{{.*}}depPattern = #arts.dep_pattern<uniform>{{.*}}planAsyncStrategy = #arts.plan_async_strategy<advance_edt>{{.*}}planRepetitionStructure = #arts.plan_repetition_structure<full_timestep>
// CHECK: arts.barrier
// CHECK-SAME: barrierReason = #arts.barrier_reason<required_memory>
// CHECK: arts.barrier
// CHECK-SAME: barrierReason = #arts.barrier_reason<timestep_stage_boundary>
// CHECK: arts.edt <task>{{.*}}depPattern = #arts.dep_pattern<jacobi_alternating_buffers>{{.*}}planAsyncStrategy = #arts.plan_async_strategy<advance_edt>{{.*}}planRepetitionStructure = #arts.plan_repetition_structure<full_timestep>
// CHECK-NOT: sde.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @jacobi_pair(%A: memref<18x18xf64>, %B: memref<18x18xf64>) {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    sde.cu_region <parallel> {
      sde.su_iterate (%c1) to (%c16) step (%c1) {
      ^bb0(%i: index):
        scf.for %j = %c1 to %c16 step %c1 {
          %v = memref.load %B[%i, %j] : memref<18x18xf64>
          memref.store %v, %A[%i, %j] : memref<18x18xf64>
        }
        sde.yield
      }
      sde.su_barrier
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
    return
  }
}
