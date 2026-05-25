// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_multinode.cfg --start-from sde-planning --pipeline codir-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// C/OpenMP often exposes only the outer parallel loop as an SDE scheduling
// dimension. For out-of-place perfect stencil nests, PatternAnalysis promotes
// the inner scf.for dimensions into the SDE owner loop so later SDE planning can
// create a true ND owner-tile plan instead of a one-dimensional owner strip.

// CHECK-LABEL: // -----// IR Dump After PatternAnalysis (sde-pattern-analysis) //----- //
// CHECK: func.func @main
// CHECK: sde.su_iterate (%c1, %c1, %c1) to (%c63, %c63, %c63) step
// CHECK-SAME: classification(<stencil>)
// CHECK: } {accessMaxOffsets = [1, 1, 1]
// CHECK-SAME: ownerDims = [0, 1, 2]
// CHECK-SAME: pattern = #sde.pattern<cross_dim_stencil_3d>
// CHECK-LABEL: // -----// IR Dump After LoopInterchange
//
// CHECK-LABEL: // -----// IR Dump After Tiling (tiling) //----- //
// CHECK: func.func @main
// CHECK: sde.su_iterate
// CHECK: } {accessMaxOffsets
// CHECK-SAME: iterationTopology = #sde.iteration_topology<owner_tile>
// CHECK-SAME: physicalHaloShape = [1, 1, 1]
// CHECK-SAME: physicalOwnerDims = [0, 1, 2]
// CHECK-LABEL: // -----// IR Dump After Vectorization

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<64x64x64xf64>, %B: memref<64x64x64xf64>) {
    %c1 = arith.constant 1 : index
    %c63 = arith.constant 63 : index
    sde.cu_region <parallel> {
      sde.su_iterate (%c1) to (%c63) step (%c1) {
      ^bb0(%i: index):
        scf.for %j = %c1 to %c63 step %c1 {
          scf.for %k = %c1 to %c63 step %c1 {
            %im1 = arith.subi %i, %c1 : index
            %ip1 = arith.addi %i, %c1 : index
            %jm1 = arith.subi %j, %c1 : index
            %jp1 = arith.addi %j, %c1 : index
            %km1 = arith.subi %k, %c1 : index
            %kp1 = arith.addi %k, %c1 : index
            %x0 = memref.load %A[%im1, %j, %k] : memref<64x64x64xf64>
            %x1 = memref.load %A[%ip1, %j, %k] : memref<64x64x64xf64>
            %y0 = memref.load %A[%i, %jm1, %k] : memref<64x64x64xf64>
            %y1 = memref.load %A[%i, %jp1, %k] : memref<64x64x64xf64>
            %z0 = memref.load %A[%i, %j, %km1] : memref<64x64x64xf64>
            %z1 = memref.load %A[%i, %j, %kp1] : memref<64x64x64xf64>
            %s0 = arith.addf %x0, %x1 : f64
            %s1 = arith.addf %y0, %y1 : f64
            %s2 = arith.addf %z0, %z1 : f64
            %s3 = arith.addf %s0, %s1 : f64
            %sum = arith.addf %s3, %s2 : f64
            memref.store %sum, %B[%i, %j, %k] : memref<64x64x64xf64>
          }
        }
        sde.yield
      }
      sde.yield
    }
    return
  }
}
