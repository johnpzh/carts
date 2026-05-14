// RUN: %carts-compile %s --O3 --arts-config %arts_config --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 \
// RUN:   | awk '/IR Dump After BarrierElimination/,/IR Dump After LowerToMemref/' \
// RUN:   | %FileCheck %s --check-prefix=SDE
// RUN: %carts-compile %s --O3 --arts-config %arts_config --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=ARTS

// Verify mixed barrier elimination with three loops under the
// carrier-authoritative model.
// Barrier 1: A writes B, B reads C — disjoint → ELIMINATED.
// Barrier 2: B writes D, C reads D — overlap → PRESERVED.

// SDE-LABEL: // -----// IR Dump After BarrierElimination (barrier-elimination) //----- //
// Barrier 1 is eliminated (disjoint), barrier 2 is preserved (overlap):
// SDE: sde.su_barrier {barrierEliminated, barrierReason = #sde.barrier_reason<redundant>}
// SDE: sde.su_barrier
// SDE-NOT: arts.barrierEliminated

// After ConvertSdeToArts: only the second barrier survives.
// ARTS-LABEL: // -----// IR Dump After ConvertSdeToArts (convert-sde-to-arts) //----- //
// ARTS: func.func @main
// Loop A:
// ARTS: arts.edt <task>
// ARTS: memref.load
// ARTS: memref.store
// No barrier between A and B (eliminated):
// Loop B:
// ARTS: arts.edt <task>
// ARTS: memref.load
// ARTS: memref.store
// Barrier between B and C (preserved):
// ARTS: arts.barrier
// Loop C:
// ARTS: arts.edt <task>
// ARTS: memref.load
// ARTS: memref.store
// ARTS-NOT: sde.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<128xf32>, %B: memref<128xf32>, %C: memref<128xf32>, %D: memref<128xf32>) {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 2.0 : f32
    omp.parallel {
      // Loop A: reads A, writes B
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c0) to (%c128) step (%c1) {
          %v = memref.load %A[%i] : memref<128xf32>
          %r = arith.addf %v, %cst : f32
          memref.store %r, %B[%i] : memref<128xf32>
          omp.yield
        }
      }
      // Barrier 1: should be ELIMINATED (A writes B, B reads C -- disjoint)
      // Loop B: reads C, writes D
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c0) to (%c128) step (%c1) {
          %v = memref.load %C[%i] : memref<128xf32>
          %r = arith.mulf %v, %cst : f32
          memref.store %r, %D[%i] : memref<128xf32>
          omp.yield
        }
      }
      // Barrier 2: should be PRESERVED (B writes D, C reads D -- overlap)
      // Loop C: reads D (written by Loop B)
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c0) to (%c128) step (%c1) {
          %v = memref.load %D[%i] : memref<128xf32>
          %r = arith.addf %v, %cst : f32
          memref.store %r, %A[%i] : memref<128xf32>
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }
}
