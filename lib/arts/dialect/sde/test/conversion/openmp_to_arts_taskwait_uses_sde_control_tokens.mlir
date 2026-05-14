// RUN: %carts-compile %s --O3 --arts-config %arts_config --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// CHECK-LABEL: // -----// IR Dump After ConvertOpenMPToSde (convert-openmp-to-sde) //----- //
// CHECK: func.func @main
// CHECK: arts_sde.cu_task
// CHECK: %[[DONE0:.*]] = arts_sde.control_token : !arts_sde.completion
// CHECK: arts_sde.cu_task
// CHECK: %[[DONE1:.*]] = arts_sde.control_token : !arts_sde.completion
// CHECK: arts_sde.su_barrier(%[[DONE0]], %[[DONE1]] : !arts_sde.completion, !arts_sde.completion)

// CHECK-LABEL: // -----// IR Dump After ConvertSdeToArts (convert-sde-to-arts) //----- //
// CHECK: func.func @main
// CHECK-NOT: arts_sde.control_token
// CHECK-NOT: arts_sde.su_barrier
// CHECK: arts.barrier

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<2xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %v0 = arith.constant 7 : i32
    %v1 = arith.constant 11 : i32
    omp.parallel {
      omp.task {
        memref.store %v0, %A[%c0] : memref<2xi32>
        omp.terminator
      }
      omp.task {
        memref.store %v1, %A[%c1] : memref<2xi32>
        omp.terminator
      }
      omp.taskwait
      omp.terminator
    }
    return
  }
}
