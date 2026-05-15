// RUN: not %carts-compile %s --O3 --arts-config %arts_config --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Verify the before/after for task dependency ownership:
// ConvertOpenMPToSde must derive SDE dependency access modes from the OpenMP
// depend clause, then materialize matching sde.mu_dep values on sde.cu_task.
// Until SDE lowers those deps into mu_token/codelet form, the boundary rejects
// the raw deps instead of letting Core rediscover task dependency ownership.

// CHECK-LABEL: // -----// IR Dump After ConvertOpenMPToSde (convert-openmp-to-sde) //----- //
// CHECK: func.func @main
// CHECK: sde.cu_region <parallel> {
// CHECK: %[[WRITEDEP:.+]] = sde.mu_dep <write> %arg0 : memref<1xi32> -> !sde.dep
// CHECK: sde.cu_task deps(%[[WRITEDEP]] : !sde.dep) {
// CHECK: %[[READDEP:.+]] = sde.mu_dep <read> %arg0 : memref<1xi32> -> !sde.dep
// CHECK: sde.cu_task deps(%[[READDEP]] : !sde.dep) {
// CHECK: has unmaterialized sde.mu_dep at the SDE/Core boundary

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<1xi32>, %B: memref<1xi32>) {
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    omp.parallel {
      omp.task depend(taskdependout -> %A : memref<1xi32>) {
        memref.store %c1_i32, %A[%c0] : memref<1xi32>
        omp.terminator
      }
      omp.task depend(taskdependin -> %A : memref<1xi32>) {
        %v = memref.load %A[%c0] : memref<1xi32>
        %sum = arith.addi %v, %c1_i32 : i32
        memref.store %sum, %B[%c0] : memref<1xi32>
        omp.terminator
      }
      omp.terminator
    }
    return
  }
}
