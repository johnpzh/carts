// RUN: %carts-compile %s --O3 --arts-config %arts_config --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Verify that Tiling leaves reduction-classified loops scalar. A pure
// reduction must not be tiled or converted to a carrier-authoritative
// linalg.generic inside su_iterate because that would duplicate the reduction
// domain per distributed iteration lane.

// CHECK-LABEL: // -----// IR Dump After Tiling (tiling) //----- //
// CHECK: func.func @main
// The reduction su_iterate must keep its original step (NOT tiled):
// CHECK: arts_sde.cu_region <parallel> {
// CHECK: arts_sde.su_iterate (%c0) to (%c128) step (%c1)
// CHECK-SAME: classification(<reduction>)
// CHECK: arts_sde.cu_region <parallel> {
// CHECK: memref.load %arg0[%{{.*}}] : memref<128xf64>
// CHECK: memref.load %{{.*}}[%c0] : memref<?xf64>
// CHECK: arith.addf
// CHECK: memref.store %{{.*}}, %{{.*}}[%c0] : memref<?xf64>
// CHECK-NOT: linalg.generic

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  omp.declare_reduction @add_f64 : f64 init {
  ^bb0(%arg0: f64):
    %cst = arith.constant 0.0 : f64
    omp.yield(%cst : f64)
  } combiner {
  ^bb0(%lhs: f64, %rhs: f64):
    %sum = arith.addf %lhs, %rhs : f64
    omp.yield(%sum : f64)
  }

  func.func @main(%A: memref<128xf64>) {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %acc = memref.alloca() : memref<1xf64>
    %acc_cast = memref.cast %acc : memref<1xf64> to memref<?xf64>
    %cst = arith.constant 0.0 : f64
    memref.store %cst, %acc[%c0] : memref<1xf64>
    omp.parallel {
      omp.wsloop reduction(@add_f64 %acc_cast -> %prv : memref<?xf64>) {
        omp.loop_nest (%i) : index = (%c0) to (%c128) step (%c1) {
          %val = memref.load %A[%i] : memref<128xf64>
          %old = memref.load %prv[%c0] : memref<?xf64>
          %new = arith.addf %old, %val : f64
          memref.store %new, %prv[%c0] : memref<?xf64>
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }
}
