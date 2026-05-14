// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from openmp-to-arts --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Verify that Tiling tiles a stencil su_iterate when
// accessMinOffsets/accessMaxOffsets are present. The tile size must be at
// least as large as the halo width (maxOffset - minOffset + 1 = 3 per dim).

// CHECK-LABEL: // -----// IR Dump After Tiling (tiling) //----- //
// CHECK: func.func @main
// The halo width is 3 (from [-1,1]), so the tile must be max(balanced, 3).
// CHECK: %[[HALO:.+]] = arith.constant 3 : index
// CHECK: arith.maxui %{{.+}}, %[[HALO]]
// CHECK: sde.su_iterate
// The outer tile step from the su_iterate feeds an scf.for tile loop:
// CHECK: scf.for
// The original loop body is cloned inside:
// CHECK: memref.load
// CHECK: memref.store

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<64x64xf64>, %B: memref<64x64xf64>) {
    %c1 = arith.constant 1 : index
    %c63 = arith.constant 63 : index
    sde.cu_region <parallel> scope(<local>) {
      sde.su_iterate (%c1) to (%c63) step (%c1) classification(<stencil>) {
      ^bb0(%i: index):
        scf.for %j = %c1 to %c63 step %c1 {
          %im1 = arith.subi %i, %c1 : index
          %ip1 = arith.addi %i, %c1 : index
          %jm1 = arith.subi %j, %c1 : index
          %jp1 = arith.addi %j, %c1 : index
          %n = memref.load %A[%im1, %j] : memref<64x64xf64>
          %s = memref.load %A[%ip1, %j] : memref<64x64xf64>
          %w = memref.load %A[%i, %jm1] : memref<64x64xf64>
          %e = memref.load %A[%i, %jp1] : memref<64x64xf64>
          %c = memref.load %A[%i, %j] : memref<64x64xf64>
          %s0 = arith.addf %n, %s : f64
          %s1 = arith.addf %w, %e : f64
          %s2 = arith.addf %s0, %s1 : f64
          %sum = arith.addf %s2, %c : f64
          memref.store %sum, %B[%i, %j] : memref<64x64xf64>
        }
        sde.yield
      } {accessMaxOffsets = [1, 1], accessMinOffsets = [-1, -1], ownerDims = [0, 1], spatialDims = [0, 1], writeFootprint = [1, 1]}
      sde.yield
    }
    return
  }
}
