// RUN: %carts-compile %s --O3 --arts-config %arts_config --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Direct memory accumulators of the form:
//   for j { C[i,j] = 0; for k { C[i,j] += A[i,k] * B[k,j]; } }
// should be distributed into an init loop followed by k-j order.
//
// CHECK-LABEL: // -----// IR Dump After LoopInterchange (loop-interchange) //----- //
// CHECK: func.func @main
// CHECK: sde.su_iterate
// CHECK: scf.for %[[INITJ:[^ ]+]] = %c0 to %c16 step %c1 {
// CHECK: memref.store %{{.*}}, %{{.*}}[%{{.*}}, %[[INITJ]]] : memref<16x16xf64>
// CHECK: scf.for %[[K:[^ ]+]] = %c0 to %c16 step %c1 {
// CHECK: scf.for %[[J:[^ ]+]] = %c0 to %c16 step %c1 {
// CHECK: memref.load %{{.*}}[%{{.*}}, %[[K]]] : memref<16x16xf64>
// CHECK: memref.load %{{.*}}[%[[K]], %[[J]]] : memref<16x16xf64>
// CHECK: memref.load %{{.*}}[%{{.*}}, %[[J]]] : memref<16x16xf64>
// CHECK: arith.addf
// CHECK: memref.store %{{.*}}, %{{.*}}[%{{.*}}, %[[J]]] : memref<16x16xf64>

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<16x16xf64>, %B: memref<16x16xf64>, %C: memref<16x16xf64>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f64
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c0) to (%c16) step (%c1) {
          scf.for %j = %c0 to %c16 step %c1 {
            memref.store %zero, %C[%i, %j] : memref<16x16xf64>
            scf.for %k = %c0 to %c16 step %c1 {
              %a = memref.load %A[%i, %k] : memref<16x16xf64>
              %b = memref.load %B[%k, %j] : memref<16x16xf64>
              %prod = arith.mulf %a, %b : f64
              %old = memref.load %C[%i, %j] : memref<16x16xf64>
              %next = arith.addf %old, %prod : f64
              memref.store %next, %C[%i, %j] : memref<16x16xf64>
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
