// RUN: %carts-compile %s --O3 --arts-config %arts_config --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Verify that structured analysis no longer bails out on stencil bodies that
// guard boundary points with an scf.if in the innermost loop body.

// CHECK-LABEL: // -----// IR Dump After StructuredSummaries (structured-summaries) //----- //
// CHECK: func.func @main
// CHECK: arts_sde.su_iterate (%c1) to (%c63) step (%c1) classification(<stencil>) {
// CHECK: } {accessMaxOffsets = [1, 1], accessMinOffsets = [-1, -1], depFamily = #arts_sde.dep_family<stencil_tiling_nd>, ownerDims = [0, 1], spatialDims = [0, 1], writeFootprint = [1, 1]}

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<64x64xf64>, %B: memref<64x64xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c63 = arith.constant 63 : index
    %c64 = arith.constant 64 : index
    %zero = arith.constant 0.0 : f64
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c1) to (%c63) step (%c1) {
          scf.for %j = %c0 to %c64 step %c1 {
            %isFirst = arith.cmpi eq, %j, %c0 : index
            %isLast = arith.cmpi eq, %j, %c63 : index
            %isBoundary = arith.ori %isFirst, %isLast : i1
            scf.if %isBoundary {
              memref.store %zero, %B[%i, %j] : memref<64x64xf64>
            } else {
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
          }
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }
}
