// Verify the full tensor-first pipeline for a scalar single+wsloop pattern:
//   1. ConvertOpenMPToSde wraps su_iterate body in cu_region <parallel>
//   2. RaiseToTensor threads scalar alloca through cu_region <single> as tensor iter_arg
//   3. ConvertToCodelet + ConvertSdeToArts preserve scalar single state on the
//      user-visible alloca and lower the loop to Core task dispatch

// RUN: %carts-compile %s --O3 --arts-config %arts_config --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 \
// RUN:   | awk '/IR Dump After ConvertOpenMPToSde/,/IR Dump After RaiseToLinalg/' \
// RUN:   | %FileCheck %s --check-prefix=OMP
// RUN: %carts-compile %s --O3 --arts-config %arts_config --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 \
// RUN:   | awk '/IR Dump After RaiseToTensor/,/IR Dump After LoopInterchange/' \
// RUN:   | %FileCheck %s --check-prefix=TENSOR
// RUN: %carts-compile %s --O3 --arts-config %arts_config --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 \
// RUN:   | awk '/IR Dump After ConvertSdeToArts/,/IR Dump After VerifySdeLowered/' \
// RUN:   | %FileCheck %s --check-prefix=ARTS

// Verify that:
//   1. ConvertOpenMPToSde wraps su_iterate body in cu_region <parallel>
//   2. RaiseToTensor threads scalar alloca through cu_region <single> as tensor iter_arg
//   3. ConvertSdeToArts produces arts.edt <single> with DB-backed memref.load/store

// --- After ConvertOpenMPToSde ---
// OMP: func.func @main
// OMP: arts_sde.cu_region <parallel>
// OMP: arts_sde.cu_region <single> scope(<local>)
// OMP: memref.load %{{.*}}[] : memref<i32>
// OMP: arith.addi
// OMP: memref.store %{{.*}}, %{{.*}}[] : memref<i32>
// OMP: arts_sde.su_barrier
// OMP: arts_sde.su_iterate
// OMP: arts_sde.cu_region <parallel>
// OMP: memref.load
// OMP: arith.muli
// OMP: memref.store

// --- After RaiseToTensor ---
// TENSOR: func.func @main
//   cu_region <parallel> threads both arrays and the scalar as iter_args:
// TENSOR: arts_sde.cu_region <parallel> iter_args(
//   cu_region <single> threads the scalar tensor:
// TENSOR: arts_sde.cu_region <single> scope(<local>) iter_args(%[[ARG:.+]] = %{{.*}} : tensor<i32>) -> (tensor<i32>)
// TENSOR: tensor.extract %[[ARG]][] : tensor<i32>
// TENSOR: arith.addi
// TENSOR: tensor.insert %{{.*}} into %[[ARG]][] : tensor<i32>
//   Barrier still present:
// TENSOR: arts_sde.su_barrier
//   su_iterate body with tensor ops:
// TENSOR: arts_sde.su_iterate
// TENSOR: tensor.extract
// TENSOR: arith.muli
// TENSOR: tensor.insert

// --- After ConvertSdeToArts ---
// ARTS: func.func @main
//   Single EDT keeps scalar state on the preserved alloca:
// ARTS: arts.edt <single>
// ARTS: memref.load %{{.*}}[] : memref<i32>
// ARTS: arith.addi %{{.*}}, %{{.*}} : i32
// ARTS: memref.store %{{.*}}, %{{.*}}[] : memref<i32>
//   Barrier between single and loop:
// ARTS: arts.barrier
//   Parallel loop:
// ARTS: arts.epoch
// ARTS: arts.edt <task>

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c100 = arith.constant 100 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c1000_i32 = arith.constant 1000 : i32
    %c2_i32 = arith.constant 2 : i32
    %alloca = memref.alloca() : memref<100xi32>
    %alloca_sum = memref.alloca() : memref<i32>
    memref.store %c0_i32, %alloca_sum[] : memref<i32>
    scf.for %i = %c0 to %c100 step %c1 {
      %iv = arith.index_cast %i : index to i32
      memref.store %iv, %alloca[%i] : memref<100xi32>
    }
    omp.parallel {
      omp.single {
        %v = memref.load %alloca_sum[] : memref<i32>
        %sum = arith.addi %v, %c1000_i32 : i32
        memref.store %sum, %alloca_sum[] : memref<i32>
        omp.terminator
      }
      omp.wsloop {
        omp.loop_nest (%j) : index = (%c0) to (%c100) step (%c1) {
          %elem = memref.load %alloca[%j] : memref<100xi32>
          %doubled = arith.muli %elem, %c2_i32 : i32
          memref.store %doubled, %alloca[%j] : memref<100xi32>
          omp.yield
        }
      }
      omp.terminator
    }
    %result = memref.load %alloca_sum[] : memref<i32>
    return %result : i32
  }
}
