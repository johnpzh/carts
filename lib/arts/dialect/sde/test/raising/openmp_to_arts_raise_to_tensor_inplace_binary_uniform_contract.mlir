// RUN: %carts-compile %s --O3 --arts-config %arts_config --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Non-stencil elementwise and matmul loops stay on the scalar SDE path.
// RaiseToLinalg stamps the structured classification, but does not create
// tensor/linalg carriers unless a later pass needs that representation.
//
// CHECK-LABEL: // -----// IR Dump After RaiseToLinalg (raise-to-linalg) //----- //
// CHECK: func.func @main
// CHECK: sde.su_iterate
// CHECK: memref.store
// CHECK-NOT: linalg.generic
// CHECK-LABEL: // -----// IR Dump After ConvertSdeToArts (convert-sde-to-arts) //----- //
// CHECK: func.func @main
// CHECK: arts.edt <task>
// CHECK: memref.store
// CHECK-NOT: sde.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<128xf64>, %B: memref<128xf64>) {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c0) to (%c128) step (%c1) {
          %a = memref.load %A[%i] : memref<128xf64>
          %b = memref.load %B[%i] : memref<128xf64>
          %sum = arith.addf %b, %a : f64
          memref.store %sum, %B[%i] : memref<128xf64>
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }
}
