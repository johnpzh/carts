// RUN: %carts-compile %s --O3 --arts-config %arts_config --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Verify that a 1D 3-point stencil B[i] = A[i-1] + A[i] + A[i+1] gets a
// shifted-view linalg.generic carrier with identity indexing maps. The scalar
// body (scf.for + memref.load/store) is preserved alongside (dual-rep).

// CHECK-LABEL: // -----// IR Dump After RaiseToLinalg (raise-to-linalg) //----- //
// CHECK: func.func @main
// CHECK: sde.su_iterate (%c1) to (%c127) step (%c1) classification(<stencil>) {
// Scalar body preserved:
// CHECK: memref.load
// CHECK: memref.store
// Carrier created with shifted extract_slice views:
// CHECK: sde.mu_memref_to_tensor %arg0 : memref<128xf64>
// CHECK: tensor.extract_slice
// CHECK: tensor.extract_slice
// CHECK: tensor.extract_slice
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel"]
// CHECK: arith.addf
// CHECK: arith.addf
// CHECK: linalg.yield
// CHECK: tensor.insert_slice

// After full pipeline: stencil contract stamped on Core task dispatch.
// CHECK: // -----// IR Dump After ConvertSdeToArts (convert-sde-to-arts) //----- //
// CHECK: arts.edt
// CHECK-SAME: depPattern = #arts.dep_pattern<stencil_tiling_nd>

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<128xf64>, %B: memref<128xf64>) {
    %c1 = arith.constant 1 : index
    %c127 = arith.constant 127 : index
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c1) to (%c127) step (%c1) {
          %im1 = arith.subi %i, %c1 : index
          %ip1 = arith.addi %i, %c1 : index
          %left = memref.load %A[%im1] : memref<128xf64>
          %center = memref.load %A[%i] : memref<128xf64>
          %right = memref.load %A[%ip1] : memref<128xf64>
          %s0 = arith.addf %left, %center : f64
          %sum = arith.addf %s0, %right : f64
          memref.store %sum, %B[%i] : memref<128xf64>
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }
}
