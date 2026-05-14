// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from openmp-to-arts --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Verify the before/after for transient tensor-carrier cleanup:
// ConvertSdeToArts should erase the carrier-authoritative tensor chain
// without leaving tensor IR in the ARTS loop.

// CHECK-LABEL: // -----// IR Dump After ConvertSdeToArts (convert-sde-to-arts) //----- //
// CHECK: func.func @main
// CHECK: arts.epoch
// CHECK: arts.edt <task>
// CHECK: memref.load
// CHECK: memref.store
// CHECK-NOT: arts_sde.mu_memref_to_tensor
// CHECK-NOT: bufferization.to_tensor
// CHECK-NOT: tensor.empty
// CHECK-NOT: tensor.cast
// CHECK-NOT: tensor.extract_slice
// CHECK-NOT: tensor.insert_slice
// CHECK-NOT: linalg.generic

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<16xf64>, %B: memref<16xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %cst = arith.constant 2.0 : f64
    arts_sde.cu_region <parallel> scope(<local>) {
      arts_sde.su_iterate (%c0) to (%c16) step (%c1) classification(<elementwise>) {
      ^bb0(%iv: index):
        %val = memref.load %A[%iv] : memref<16xf64>
        %mul = arith.mulf %val, %cst : f64
        memref.store %mul, %B[%iv] : memref<16xf64>

        %input = arts_sde.mu_memref_to_tensor %A : memref<16xf64> -> tensor<16xf64>
        %init = tensor.empty() : tensor<16xf64>
        %generic = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%input : tensor<16xf64>) outs(%init : tensor<16xf64>) {
        ^bb0(%in: f64, %out: f64):
          %scaled = arith.mulf %in, %cst : f64
          linalg.yield %scaled : f64
        } -> tensor<16xf64>
        %cast = tensor.cast %generic : tensor<16xf64> to tensor<16xf64>
        %slice = tensor.extract_slice %cast[0] [16] [1] : tensor<16xf64> to tensor<16xf64>
        %insert = tensor.insert_slice %slice into %init[0] [16] [1] : tensor<16xf64> into tensor<16xf64>
        arts_sde.yield
      }
      arts_sde.yield
    }
    return
  }
}
