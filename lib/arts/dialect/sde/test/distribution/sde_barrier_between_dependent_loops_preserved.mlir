// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from sde-planning --pipeline codir-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=SDE
// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from sde-planning --pipeline codir-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=ARTS

// Verify that BarrierElimination preserves a barrier when the successor
// reads from a memref that the predecessor writes to.

// SDE-LABEL: // -----// IR Dump After ConvertOpenMPToSde (convert-openmp-to-sde) //----- //
// SDE: sde.su_iterate
// SDE: sde.su_barrier
// SDE: sde.su_iterate

// SDE-LABEL: // -----// IR Dump After BarrierElimination (barrier-elimination) //----- //
// SDE: sde.su_barrier
// SDE-NOT: barrierEliminated
// SDE: sde.su_iterate
// SDE-LABEL: func.func @write_after_write_preserved
// SDE: sde.su_iterate
// SDE: sde.su_barrier
// SDE-NOT: barrierEliminated
// SDE: sde.su_iterate

// After boundary materialization, downstream synchronization must be present.
// ARTS-LABEL: // -----// IR Dump After ConvertCodirToArts (convert-codir-to-arts) //----- //
// ARTS: func.func @main
// ARTS: arts.barrier
// ARTS-LABEL: func.func @write_after_write_preserved
// ARTS: arts.barrier
// ARTS-NOT: sde.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<128xf32>, %B: memref<128xf32>) {
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
      // Loop B: reads B (written by Loop A) -- NOT disjoint
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c0) to (%c128) step (%c1) {
          %v = memref.load %B[%i] : memref<128xf32>
          %r = arith.mulf %v, %cst : f32
          memref.store %r, %A[%i] : memref<128xf32>
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }

  func.func @write_after_write_preserved(%A: memref<128xf32>, %B: memref<128xf32>) {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 2.0 : f32
    omp.parallel {
      // Loop A writes B.
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c0) to (%c128) step (%c1) {
          %v = memref.load %A[%i] : memref<128xf32>
          %r = arith.addf %v, %cst : f32
          memref.store %r, %B[%i] : memref<128xf32>
          omp.yield
        }
      }
      // Loop B also writes B. The barrier orders the final value; removing it
      // would make the two writer loops race.
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c0) to (%c128) step (%c1) {
          %v = memref.load %A[%i] : memref<128xf32>
          %r = arith.mulf %v, %cst : f32
          memref.store %r, %B[%i] : memref<128xf32>
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }
}
