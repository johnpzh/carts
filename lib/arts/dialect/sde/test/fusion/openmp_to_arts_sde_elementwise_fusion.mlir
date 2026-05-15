// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from sde-planning --pipeline codir-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=SDE
// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from sde-planning --pipeline codir-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=ARTS

// Verify that two consecutive elementwise loops with disjoint write-sets
// are fused into a single elementwise_pipeline loop under the
// carrier-authoritative model.

// Before fusion: two separate elementwise su_iterate ops.
// SDE-LABEL: // -----// IR Dump After PatternAnalysis (sde-pattern-analysis) //----- //
// SDE: func.func @main
// SDE: sde.cu_region <parallel> {
// SDE-COUNT-2: sde.su_iterate (%c0) to (%c128) step (%{{.+}}) nowait classification(<elementwise>) {

// After fusion: a single elementwise_pipeline su_iterate op.
// SDE-LABEL: // -----// IR Dump After ElementwiseFusion (elementwise-fusion) //----- //
// SDE: func.func @main
// SDE: sde.su_iterate (%c0) to (%c128) step (%{{.+}}) nowait classification(<elementwise_pipeline>) {
// SDE-NOT: classification(<elementwise>)

// After ConvertCodirToArts: single fused Core task dispatch with both computations.
// ARTS-LABEL: // -----// IR Dump After ConvertCodirToArts (convert-codir-to-arts) //----- //
// ARTS: func.func @main
// ARTS: arts.edt <task>
// ARTS: arts.edt <task>
// ARTS-SAME: arts.pattern_revision = 1 : i64
// ARTS-SAME: depPattern = #arts.dep_pattern<elementwise_pipeline>
// ARTS: memref.load
// ARTS: arith.addf
// ARTS: memref.store
// ARTS: memref.load
// ARTS: arith.mulf
// ARTS: memref.store
// ARTS-NOT: sde.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<128xf32>, %B: memref<128xf32>, %C: memref<128xf32>) {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 2.0 : f32
    omp.parallel {
      omp.wsloop nowait {
        omp.loop_nest (%i) : index = (%c0) to (%c128) step (%c1) {
          %v = memref.load %A[%i] : memref<128xf32>
          %r = arith.addf %v, %cst : f32
          memref.store %r, %B[%i] : memref<128xf32>
          omp.yield
        }
      }
      omp.wsloop nowait {
        omp.loop_nest (%i) : index = (%c0) to (%c128) step (%c1) {
          %v = memref.load %B[%i] : memref<128xf32>
          %r = arith.mulf %v, %cst : f32
          memref.store %r, %C[%i] : memref<128xf32>
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }
}
