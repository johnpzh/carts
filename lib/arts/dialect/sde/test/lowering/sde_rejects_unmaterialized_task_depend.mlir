// RUN: not %carts-compile %s --O3 --arts-config %arts_config \
// RUN:   --pipeline openmp-to-arts 2>&1 | %FileCheck %s

// CHECK: has unmaterialized sde.mu_dep at the SDE/Core boundary

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @task_depend_arg(%A: memref<1xi32>) {
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
