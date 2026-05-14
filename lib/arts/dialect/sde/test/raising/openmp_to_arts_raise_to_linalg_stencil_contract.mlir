// RUN: %carts-compile %s --O3 --arts-config %arts_config --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=OPENMP

// Verify the stencil classification and carrier raising path in RaiseToLinalg:
// after RaiseToLinalg, the su_iterate carries both the scalar body (scf.for +
// memref.load/store) and a dual-rep linalg.generic carrier with shifted
// extract_slice views. After ConvertSdeToArts, the stencil contract
// attributes land on the ARTS edt/for.

// OPENMP-LABEL: // -----// IR Dump After RaiseToLinalg (raise-to-linalg) //----- //
// OPENMP: func.func @main
// OPENMP: arts_sde.cu_region <parallel> scope(<local>) {
// OPENMP: arts_sde.su_iterate (%c1) to (%c63) step (%c1) classification(<stencil>) {
// OPENMP: arts_sde.cu_region <parallel> scope(<local>) {
// Scalar body preserved (dual-rep):
// OPENMP: scf.for %[[J:.*]] = %c1 to %c63 step %c1
// OPENMP: memref.load %arg0[%{{.*}}, %[[J]]] : memref<64x64xf64>
// OPENMP: memref.store %{{.*}}, %arg1[%{{.*}}, %[[J]]] : memref<64x64xf64>
// Shifted-view carrier alongside:
// OPENMP: arts_sde.mu_memref_to_tensor %arg0 : memref<64x64xf64>
// OPENMP: tensor.extract_slice
// OPENMP: linalg.generic
// OPENMP-SAME: iterator_types = ["parallel", "parallel"]
// OPENMP: tensor.insert_slice
// OPENMP-NOT: arts.for
// OPENMP: // -----// IR Dump After ConvertSdeToArts (convert-sde-to-arts) //----- //
// OPENMP: arts.epoch
// OPENMP-SAME: depPattern = #arts.dep_pattern<stencil_tiling_nd>
// OPENMP-SAME: planOwnerDims = [0]
// OPENMP-SAME: planPhysicalBlockShape = [4, 64]
// OPENMP-SAME: stencil_block_shape = [4, 64]
// OPENMP-SAME: stencil_max_offsets = [1, 1]
// OPENMP-SAME: stencil_min_offsets = [-1, -1]

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
