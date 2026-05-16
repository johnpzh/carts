// RUN: %carts-compile %s --pass-pipeline='builtin.module(convert-sde-to-codir,verify-codir,convert-codir-to-arts)' \
// RUN:   | %FileCheck %s --implicit-check-not=sde. --implicit-check-not=codir.codelet

// Scheduling units must be represented as CODIR codelets before ARTS
// materialization. The ARTS stage should see explicit dependencies and params,
// not residual SDE scheduling ops.

// CHECK-LABEL: func.func @main
// CHECK: arts.db_alloc
// CHECK: arts.db_alloc
// CHECK: scf.for
// CHECK: arts.db_acquire[<in>]
// CHECK: arts.db_acquire[<out>]
// CHECK: arts.edt <task> <intranode>
// CHECK-SAME: params(
// CHECK: arts.db_ref
// CHECK: arts.db_ref
// CHECK: memref.load
// CHECK: memref.store
// CHECK: arts.barrier

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<16xf64>, %B: memref<16xf64>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    sde.cu_region <parallel> {
      sde.su_distribute <cyclic> {
        sde.su_iterate (%c0) to (%c16) step (%c1) classification(<elementwise>) {
        ^bb0(%i: index):
          %v = memref.load %A[%i] : memref<16xf64>
          memref.store %v, %B[%i] : memref<16xf64>
          sde.yield
        }
      }
      sde.yield
    }
    return
  }
}
