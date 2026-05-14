// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from openmp-to-arts --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Jacobi benchmark loops lower as adjacent SDE stages in an outer timestep
// loop, with the second phase guarded by the timestep IV rather than separated
// by an explicit SDE barrier. Same-shape stencil/stencil adjacent pairs keep
// the regular stencil plan only; the repeated-timestep async contract is not
// stamped until Core has a profitable multi-stage stencil schedule for this
// shape.

// CHECK-LABEL: // -----// IR Dump After BarrierElimination (barrier-elimination) //----- //
// CHECK: func.func @jacobi_if_guarded_pair
// CHECK: scf.for
// CHECK: sde.su_iterate
// CHECK-SAME: classification(<stencil>)
// CHECK: } {
// CHECK-SAME: pattern = #sde.pattern<stencil_tiling_nd>
// CHECK-NOT: asyncStrategy
// CHECK-NOT: repetitionStructure
// CHECK: scf.if
// CHECK: sde.su_iterate
// CHECK-SAME: classification(<stencil>)
// CHECK: } {
// CHECK-SAME: pattern = #sde.pattern<stencil_tiling_nd>
// CHECK-NOT: asyncStrategy
// CHECK-NOT: repetitionStructure
// CHECK-LABEL: // -----// IR Dump After ConvertSdeToArts (convert-sde-to-arts) //----- //
// CHECK: func.func @jacobi_if_guarded_pair
// CHECK: depPattern = #arts.dep_pattern<stencil_tiling_nd>
// CHECK-NOT: planAsyncStrategy
// CHECK-NOT: planRepetitionStructure
// CHECK: scf.if
// CHECK: depPattern = #arts.dep_pattern<stencil_tiling_nd>
// CHECK-NOT: planAsyncStrategy
// CHECK-NOT: planRepetitionStructure
// CHECK-NOT: sde.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @jacobi_if_guarded_pair(%A: memref<18x18xf64>, %B: memref<18x18xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c16 = arith.constant 16 : index
    %c20 = arith.constant 20 : index
    scf.for %t = %c0 to %c20 step %c2 {
      sde.cu_region <parallel> scope(<local>) {
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
        sde.cu_region <parallel> scope(<local>) {
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
