// RUN: %carts-compile %s --O3 --arts-config %arts_config --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Verify that a 2D 5-point Jacobi stencil gets a shifted-view linalg.generic
// carrier with 5 inputs (one per neighbor access) and identity indexing maps.
// The scalar body is preserved alongside (dual-rep). After ConvertSdeToArts,
// the stencil contract attributes (min/max offsets) are stamped.

// CHECK-LABEL: // -----// IR Dump After RaiseToLinalg (raise-to-linalg) //----- //
// CHECK: func.func @main
// CHECK: arts_sde.su_iterate (%c1) to (%c63) step (%c1) classification(<stencil>) {
// Scalar body preserved:
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.store
// 5 shifted-view inputs (A[i-1,j], A[i+1,j], A[i,j-1], A[i,j+1], A[i,j]):
// CHECK: arts_sde.mu_memref_to_tensor %arg0 : memref<64x64xf64>
// CHECK-COUNT-5: tensor.extract_slice
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel"]
// 4 adds for the 5-point stencil computation:
// CHECK: arith.addf
// CHECK: arith.addf
// CHECK: arith.addf
// CHECK: arith.addf
// CHECK: linalg.yield
// CHECK: tensor.insert_slice

// After ConvertSdeToArts: stencil contract with spatial metadata.
// CHECK: // -----// IR Dump After ConvertSdeToArts (convert-sde-to-arts) //----- //
// CHECK: arts.edt <parallel> <intranode>
// CHECK-SAME: arts.plan.owner_dims = [0]
// CHECK-SAME: arts.plan.physical_block_shape = [8, 64]
// CHECK-SAME: depPattern = #arts.dep_pattern<stencil_tiling_nd>
// CHECK-SAME: stencil_block_shape = [8]
// CHECK-SAME: stencil_max_offsets = [1]
// CHECK-SAME: stencil_min_offsets = [-1]

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<64x64xf64>, %B: memref<64x64xf64>) {
    %c1 = arith.constant 1 : index
    %c63 = arith.constant 63 : index
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c1) to (%c63) step (%c1) {
          scf.for %j = %c1 to %c63 step %c1 {
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
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }
}
