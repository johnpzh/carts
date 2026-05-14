// RUN: %carts-compile %s --O3 --arts-config %arts_config --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=SDE
// RUN: %carts-compile %s --O3 --arts-config %arts_config --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=ARTS

// Verify that BarrierElimination eliminates the barrier between two loops
// with disjoint write/read sets under the carrier-authoritative model.
// Loop A: reads A, writes B.  Loop B: reads C, writes D.
// Pred writes {B}, succ reads {C} — disjoint, so barrier is eliminated.

// SDE-LABEL: // -----// IR Dump After ConvertOpenMPToSde (convert-openmp-to-sde) //----- //
// SDE: arts_sde.su_iterate
// SDE: arts_sde.su_barrier
// SDE: arts_sde.su_iterate

// SDE-LABEL: // -----// IR Dump After BarrierElimination (barrier-elimination) //----- //
// SDE: arts_sde.su_barrier {barrierEliminated, barrierReason = #arts_sde.barrier_reason<redundant>}
// SDE: arts_sde.su_iterate
// SDE-LABEL: func.func @write_only_successor_disjoint
// SDE: arts_sde.su_iterate
// SDE: arts_sde.su_barrier {barrierEliminated, barrierReason = #arts_sde.barrier_reason<redundant>}
// SDE: arts_sde.su_iterate

// After ConvertSdeToArts, the eliminated barrier is NOT lowered to arts.barrier
// ARTS-LABEL: // -----// IR Dump After ConvertSdeToArts (convert-sde-to-arts) //----- //
// ARTS: func.func @main
// ARTS: arts.edt <task>
// ARTS-NOT: arts.barrier
// ARTS: arts.edt <task>
// ARTS-LABEL: func.func @write_only_successor_disjoint
// ARTS: arts.edt <task>
// ARTS-NOT: arts.barrier
// ARTS: arts.edt <task>
// ARTS-NOT: arts_sde.

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
      // Implicit barrier here (no nowait on Loop A)
      // Loop B: reads C, writes D -- disjoint from Loop A's writes
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c0) to (%c128) step (%c1) {
          %v = memref.load %C[%i] : memref<128xf32>
          %r = arith.mulf %v, %cst : f32
          memref.store %r, %D[%i] : memref<128xf32>
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }

  func.func @write_only_successor_disjoint(%A: memref<128xf32>, %B: memref<128xf32>, %C: memref<128xf32>, %D: memref<128xf32>) {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 2.0 : f32
    omp.parallel {
      // Loop A: reads A, writes B.
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c0) to (%c128) step (%c1) {
          %v = memref.load %A[%i] : memref<128xf32>
          %r = arith.addf %v, %cst : f32
          memref.store %r, %B[%i] : memref<128xf32>
          omp.yield
        }
      }
      // Loop B only writes D. The old analysis saw an empty successor read set
      // and kept the barrier; the full read/write conflict check can remove it.
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c0) to (%c128) step (%c1) {
          memref.store %cst, %D[%i] : memref<128xf32>
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }
}
