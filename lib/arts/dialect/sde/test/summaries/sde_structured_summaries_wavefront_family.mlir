// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from openmp-to-arts --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// SDE owns the wavefront dependency family. Core only receives the translated
// family after ConvertSdeToArts.

// CHECK-LABEL: // -----// IR Dump After StructuredSummaries (structured-summaries) //----- //
// CHECK: func.func @wavefront
// CHECK: arts_sde.su_iterate (%c1, %c1) to (%c16, %c16) step (%c1, %c1) classification(<stencil>) {
// CHECK: } {
// CHECK-SAME: accessMaxOffsets = [0, 0]
// CHECK-SAME: accessMinOffsets = [-1, -1]
// CHECK-SAME: depFamily = #arts_sde.dep_family<wavefront_2d>
// CHECK-SAME: ownerDims = [0, 1]
// CHECK-SAME: writeFootprint = [1, 1]
// CHECK-LABEL: // -----// IR Dump After BarrierElimination (barrier-elimination) //----- //
// CHECK: func.func @wavefront
// CHECK: arts_sde.su_barrier
// CHECK-SAME: barrierReason = #arts_sde.barrier_reason<wavefront_frontier>
// CHECK-LABEL: // -----// IR Dump After ConvertSdeToArts (convert-sde-to-arts) //----- //
// CHECK: func.func @wavefront
// CHECK: arts.edt <parallel> <intranode> route(%{{.*}}) attributes {
// CHECK-SAME: no_verify = #arts.no_verify
// CHECK: depPattern = #arts.dep_pattern<wavefront_2d>
// CHECK: arts.barrier
// CHECK-NOT: arts_sde.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @wavefront(%A: memref<18x18xf64>) {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    arts_sde.cu_region <parallel> scope(<local>) {
      arts_sde.su_iterate (%c1, %c1) to (%c16, %c16) step (%c1, %c1) {
      ^bb0(%i: index, %j: index):
        %im1 = arith.subi %i, %c1 : index
        %jm1 = arith.subi %j, %c1 : index
        %top = memref.load %A[%im1, %j] : memref<18x18xf64>
        %left = memref.load %A[%i, %jm1] : memref<18x18xf64>
        %sum = arith.addf %top, %left : f64
        memref.store %sum, %A[%i, %j] : memref<18x18xf64>
        arts_sde.yield
      }
      arts_sde.su_barrier
      arts_sde.su_iterate (%c1, %c1) to (%c16, %c16) step (%c1, %c1) {
      ^bb0(%i: index, %j: index):
        %im1 = arith.subi %i, %c1 : index
        %jm1 = arith.subi %j, %c1 : index
        %top = memref.load %A[%im1, %j] : memref<18x18xf64>
        %left = memref.load %A[%i, %jm1] : memref<18x18xf64>
        %sum = arith.addf %top, %left : f64
        memref.store %sum, %A[%i, %j] : memref<18x18xf64>
        arts_sde.yield
      }
      arts_sde.yield
    }
    return
  }
}
