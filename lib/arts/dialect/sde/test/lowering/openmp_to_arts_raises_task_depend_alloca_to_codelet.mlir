// RUN: %carts-compile %s --O3 --arts-config %arts_config --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 \
// RUN:   | awk '/IR Dump After RaiseMemrefToTensor/,/IR Dump After RaiseToTensor/' \
// RUN:   | %FileCheck %s --check-prefix=SDE
// RUN: %carts-compile %s --O3 --arts-config %arts_config --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 \
// RUN:   | awk '/IR Dump After ConvertSdeToArts/,/IR Dump After VerifySdeLowered/' \
// RUN:   | %FileCheck %s --check-prefix=CORE

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  // SDE-LABEL: func.func @raised_task_depend_alloca
  // SDE: sde.mu_data shared : tensor<2xi32>
  // SDE: sde.mu_token <write>
  // SDE: sde.cu_codelet
  // SDE: sde.mu_token <readwrite>
  // SDE: sde.cu_codelet
  // SDE-NOT: sde.cu_task
  //
  // CORE-LABEL: func.func @raised_task_depend_alloca
  // CORE: arts.db_alloc
  // CORE-DAG: arts.db_acquire[<out>]
  // CORE-DAG: arts.db_acquire[<inout>]
  // CORE-NOT: arts.db_control
  func.func @raised_task_depend_alloca() {
    %A = memref.alloca() : memref<2xi32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %v0 = arith.constant 7 : i32
    %v1 = arith.constant 1 : i32
    omp.parallel {
      omp.task depend(taskdependout -> %A : memref<2xi32>) {
        memref.store %v0, %A[%c0] : memref<2xi32>
        omp.terminator
      }
      omp.task depend(taskdependin -> %A : memref<2xi32>) {
        %loaded = memref.load %A[%c0] : memref<2xi32>
        %sum = arith.addi %loaded, %v1 : i32
        memref.store %sum, %A[%c1] : memref<2xi32>
        omp.terminator
      }
      omp.terminator
    }
    return
  }

  // SDE-LABEL: func.func @fallback_task_depend_arg
  // SDE: sde.mu_dep <write> %arg0
  // SDE: sde.cu_task
  //
  // CORE: func.func @fallback_task_depend_arg
  // CORE: arts.db_control[<out>]
  func.func @fallback_task_depend_arg(%A: memref<1xi32>) {
    %c0 = arith.constant 0 : index
    %v0 = arith.constant 11 : i32
    omp.parallel {
      omp.task depend(taskdependout -> %A : memref<1xi32>) {
        memref.store %v0, %A[%c0] : memref<1xi32>
        omp.terminator
      }
      omp.terminator
    }
    return
  }
}
