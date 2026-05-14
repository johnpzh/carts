// RUN: %carts-compile %s --O3 --arts-config %arts_config --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Verify that ElementwiseFusion does NOT fuse two consecutive elementwise
// loops when they both write to the same memref (%B), i.e. the write sets
// overlap.

// CHECK-LABEL: // -----// IR Dump After ElementwiseFusion (elementwise-fusion) //----- //
// CHECK: func.func @main
// CHECK-COUNT-2: sde.su_iterate
// CHECK-NOT: elementwise_pipeline

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<128xf32>, %B: memref<128xf32>) {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %cst1 = arith.constant 2.0 : f32
    %cst2 = arith.constant 3.0 : f32
    omp.parallel {
      omp.wsloop nowait {
        omp.loop_nest (%i) : index = (%c0) to (%c128) step (%c1) {
          %v = memref.load %A[%i] : memref<128xf32>
          %r = arith.addf %v, %cst1 : f32
          memref.store %r, %B[%i] : memref<128xf32>
          omp.yield
        }
      }
      omp.wsloop nowait {
        omp.loop_nest (%i) : index = (%c0) to (%c128) step (%c1) {
          %v = memref.load %A[%i] : memref<128xf32>
          %r = arith.mulf %v, %cst2 : f32
          memref.store %r, %B[%i] : memref<128xf32>
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }
}
