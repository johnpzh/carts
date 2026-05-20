// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from sde-planning --pipeline codir-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Verify that a 1D 3-point stencil B[i] = A[i-1] + A[i] + A[i+1] stays
// memref-native and is classified with stencil access metadata. The scalar body
// (memref.load/store) is preserved as the owning representation.

// CHECK-LABEL: // -----// IR Dump After PatternAnalysis (sde-pattern-analysis) //----- //
// CHECK: func.func @main
// CHECK: sde.su_iterate (%c1) to (%c127) step (%c1) classification(<stencil>) {
// Scalar body preserved:
// CHECK: memref.load
// CHECK: memref.store
// Stencil access metadata stamped by PatternAnalysis:
// CHECK: accessMaxOffsets = [1]
// CHECK-SAME: accessMinOffsets = [-1]
// CHECK-SAME: ownerDims = [0]
// CHECK-SAME: pattern = #sde.pattern<stencil_tiling_nd>
// CHECK-SAME: spatialDims = [0]
// CHECK-SAME: writeFootprint = [1]
// CHECK: arith.addf
// CHECK: arith.addf

// After full pipeline: stencil contract stamped on ARTS task dispatch.
// CHECK: // -----// IR Dump After ConvertCodirToArts (convert-codir-to-arts) //----- //
// CHECK: arts.edt <task> {{.*}}depPattern = #arts.dep_pattern<stencil_tiling_nd>

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<128xf64>, %B: memref<128xf64>) {
    %c1 = arith.constant 1 : index
    %c127 = arith.constant 127 : index
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c1) to (%c127) step (%c1) {
          %im1 = arith.subi %i, %c1 : index
          %ip1 = arith.addi %i, %c1 : index
          %left = memref.load %A[%im1] : memref<128xf64>
          %center = memref.load %A[%i] : memref<128xf64>
          %right = memref.load %A[%ip1] : memref<128xf64>
          %s0 = arith.addf %left, %center : f64
          %sum = arith.addf %s0, %right : f64
          memref.store %sum, %B[%i] : memref<128xf64>
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }
}
