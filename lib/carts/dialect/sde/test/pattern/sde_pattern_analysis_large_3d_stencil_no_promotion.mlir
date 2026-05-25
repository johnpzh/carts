// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_multinode.cfg --start-from sde-planning --pipeline codir-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Very large static 3D stencils stay on the existing outer-owner path. The
// unrestricted ND promotion is semantically valid but can make LLVM backend
// compilation dominate benchmark build time for large production domains.

// CHECK-LABEL: // -----// IR Dump After PatternAnalysis (sde-pattern-analysis) //----- //
// CHECK: func.func @main
// CHECK: sde.su_iterate (%c1) to (%c511) step
// CHECK-SAME: classification(<stencil>)
// CHECK-NOT: sde.su_iterate (%c1, %c1, %c1)
// CHECK: } {accessMaxOffsets = [1, 1, 1]
// CHECK-SAME: ownerDims = [0, 1, 2]
// CHECK-SAME: pattern = #sde.pattern<cross_dim_stencil_3d>
// CHECK-LABEL: // -----// IR Dump After LoopInterchange

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<512x512x512xf64>, %B: memref<512x512x512xf64>) {
    %c1 = arith.constant 1 : index
    %c511 = arith.constant 511 : index
    sde.cu_region <parallel> {
      sde.su_iterate (%c1) to (%c511) step (%c1) {
      ^bb0(%i: index):
        scf.for %j = %c1 to %c511 step %c1 {
          scf.for %k = %c1 to %c511 step %c1 {
            %im1 = arith.subi %i, %c1 : index
            %ip1 = arith.addi %i, %c1 : index
            %jm1 = arith.subi %j, %c1 : index
            %jp1 = arith.addi %j, %c1 : index
            %km1 = arith.subi %k, %c1 : index
            %kp1 = arith.addi %k, %c1 : index
            %x0 = memref.load %A[%im1, %j, %k] : memref<512x512x512xf64>
            %x1 = memref.load %A[%ip1, %j, %k] : memref<512x512x512xf64>
            %y0 = memref.load %A[%i, %jm1, %k] : memref<512x512x512xf64>
            %y1 = memref.load %A[%i, %jp1, %k] : memref<512x512x512xf64>
            %z0 = memref.load %A[%i, %j, %km1] : memref<512x512x512xf64>
            %z1 = memref.load %A[%i, %j, %kp1] : memref<512x512x512xf64>
            %s0 = arith.addf %x0, %x1 : f64
            %s1 = arith.addf %y0, %y1 : f64
            %s2 = arith.addf %z0, %z1 : f64
            %s3 = arith.addf %s0, %s1 : f64
            %sum = arith.addf %s3, %s2 : f64
            memref.store %sum, %B[%i, %j, %k] : memref<512x512x512xf64>
          }
        }
        sde.yield
      }
      sde.yield
    }
    return
  }
}
