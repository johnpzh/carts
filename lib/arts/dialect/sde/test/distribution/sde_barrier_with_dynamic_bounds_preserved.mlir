// RUN: %carts-compile %s --O3 --arts-config %arts_config --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=SDE
// RUN: %carts-compile %s --O3 --arts-config %arts_config --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=ARTS

// Verify that BarrierElimination preserves barriers when loops have dynamic
// bounds AND overlapping memory (predecessor writes %A, successor reads %A).
// Dynamic bounds do not prevent write/read set analysis -- the barrier is kept
// because the sets overlap, not because bounds are unknown.

// SDE-LABEL: // -----// IR Dump After ConvertOpenMPToSde (convert-openmp-to-sde) //----- //
// SDE: sde.su_iterate
// SDE: sde.su_barrier
// SDE: sde.su_iterate

// SDE-LABEL: // -----// IR Dump After BarrierElimination (barrier-elimination) //----- //
// SDE: sde.su_barrier
// SDE-NOT: barrierEliminated
// SDE: sde.su_iterate

// After ConvertSdeToArts, arts.barrier must be present
// ARTS-LABEL: // -----// IR Dump After ConvertSdeToArts (convert-sde-to-arts) //----- //
// ARTS: func.func @main
// ARTS: arts.barrier
// ARTS-NOT: sde.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<?xf32>, %N: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 2.0 : f32
    omp.parallel {
      // Loop 1: reads A, writes A (in-place scale)
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c0) to (%N) step (%c1) {
          %v = memref.load %A[%i] : memref<?xf32>
          %r = arith.mulf %v, %cst : f32
          memref.store %r, %A[%i] : memref<?xf32>
          omp.yield
        }
      }
      // Loop 2: reads A (depends on Loop 1's writes)
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c0) to (%N) step (%c1) {
          %v = memref.load %A[%i] : memref<?xf32>
          %r = arith.addf %v, %cst : f32
          memref.store %r, %A[%i] : memref<?xf32>
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }
}
