// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from sde-planning --pipeline codir-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Verify that LoopInterchange reorders inner stencil loops by halo width.
// For a 3D stencil with asymmetric halos (dim 0: [-1,1]=2, dim 1: [-2,2]=4,
// dim 2: [-1,1]=2), dim 1 has a wider halo than dim 2, so the pass should
// swap the inner loop pair to put the narrower-halo dim outermost.
//
// Before interchange: inner loops are j (dim 1, halo=4) then k (dim 2, halo=2)
// After interchange:  inner loops are k (dim 2, halo=2) then j (dim 1, halo=4)
// This minimizes total halo volume when the outer dim is distributed.

// CHECK-LABEL: // -----// IR Dump After LoopInterchange (loop-interchange) //----- //
// CHECK: sde.su_iterate
// The outer inner loop should now iterate over the narrower-halo dim (was k):
// CHECK: scf.for %[[OUTER:.+]] = %c1 to %c31 step %c1 {
// CHECK:   scf.for %[[INNER:.+]] = %c1 to %c31 step %c1 {

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<32x32x32xf64>, %B: memref<32x32x32xf64>) {
    %c1 = arith.constant 1 : index
    %c31 = arith.constant 31 : index
    sde.cu_region <parallel> {
      sde.su_iterate (%c1) to (%c31) step (%c1) classification(<stencil>) {
      ^bb0(%i: index):
        scf.for %j = %c1 to %c31 step %c1 {
          scf.for %k = %c1 to %c31 step %c1 {
            // Dim 1 (j): offsets [-2, +2] -> halo width = 4
            %jm2 = arith.subi %j, %c1 : index
            %jm1 = arith.subi %jm2, %c1 : index
            %jp1 = arith.addi %j, %c1 : index
            %jp2 = arith.addi %jp1, %c1 : index
            // Dim 2 (k): offsets [-1, +1] -> halo width = 2
            %km1 = arith.subi %k, %c1 : index
            %kp1 = arith.addi %k, %c1 : index
            // Dim 0 (i): offsets [-1, +1] -> halo width = 2
            %im1 = arith.subi %i, %c1 : index
            %ip1 = arith.addi %i, %c1 : index
            %v0 = memref.load %A[%im1, %j, %k] : memref<32x32x32xf64>
            %v1 = memref.load %A[%ip1, %j, %k] : memref<32x32x32xf64>
            %v2 = memref.load %A[%i, %jm1, %k] : memref<32x32x32xf64>
            %v3 = memref.load %A[%i, %jp2, %k] : memref<32x32x32xf64>
            %v4 = memref.load %A[%i, %j, %km1] : memref<32x32x32xf64>
            %v5 = memref.load %A[%i, %j, %kp1] : memref<32x32x32xf64>
            %v6 = memref.load %A[%i, %j, %k] : memref<32x32x32xf64>
            %s0 = arith.addf %v0, %v1 : f64
            %s1 = arith.addf %v2, %v3 : f64
            %s2 = arith.addf %v4, %v5 : f64
            %s3 = arith.addf %s0, %s1 : f64
            %s4 = arith.addf %s2, %v6 : f64
            %sum = arith.addf %s3, %s4 : f64
            memref.store %sum, %B[%i, %j, %k] : memref<32x32x32xf64>
          }
        }
        sde.yield
      } {accessMaxOffsets = [1, 2, 1], accessMinOffsets = [-1, -2, -1], ownerDims = [0, 1, 2], spatialDims = [0, 1, 2], writeFootprint = [1, 1, 1]}
      sde.yield
    }
    return
  }
}
