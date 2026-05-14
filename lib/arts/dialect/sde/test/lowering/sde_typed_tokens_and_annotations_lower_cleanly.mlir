// RUN: %carts-compile %s --O3 --arts-config %arts_config --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Verify that the typed SDE dep/completion surfaces print at the SDE boundary
// and that the new annotation-only SDE ops lower away cleanly before
// VerifySdeLowered.

// CHECK-LABEL: // -----// IR Dump After ConvertOpenMPToSde (convert-openmp-to-sde) //----- //
// CHECK: arts_sde.mu_reduction_decl @sum_i32 : i32 kind(<add>) identity(0 : i32)
// CHECK: func.func @typed_tokens(%arg0: memref<8xi32>, %arg1: !arts_sde.dep, %arg2: !arts_sde.completion)
// CHECK: arts_sde.cu_task deps(%arg1 : !arts_sde.dep) {
// CHECK: memref.store %c0_i32, %arg0[%c0] : memref<8xi32>
// CHECK: arts_sde.su_barrier(%arg2 : !arts_sde.completion)

// CHECK-LABEL: // -----// IR Dump After ConvertSdeToArts (convert-sde-to-arts) //----- //
// CHECK: func.func @typed_tokens(%arg0: memref<8xi32>, %arg1: !arts_sde.dep, %arg2: !arts_sde.completion)
// CHECK-NOT: arts_sde.mu_reduction_decl
// CHECK-NOT: arts_sde.cu_region
// CHECK-NOT: arts_sde.cu_task
// CHECK: arts.edt <task>
// CHECK: arts.barrier

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  arts_sde.mu_reduction_decl @sum_i32 : i32 kind(<add>) identity(0 : i32)

  func.func @typed_tokens(%A: memref<8xi32>, %dep: !arts_sde.dep, %done: !arts_sde.completion) {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32

    arts_sde.cu_region <parallel> scope(<local>) {
      arts_sde.cu_task deps(%dep : !arts_sde.dep) {
        memref.store %c0_i32, %A[%c0] : memref<8xi32>
        arts_sde.yield
      }
      arts_sde.su_barrier(%done : !arts_sde.completion)
      arts_sde.yield
    }
    return
  }
}
