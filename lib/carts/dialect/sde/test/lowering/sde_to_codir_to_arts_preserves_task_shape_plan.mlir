// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from sde-planning --pipeline codir-to-arts \
// RUN:   | %FileCheck %s --implicit-check-not=sde. --implicit-check-not=codir.codelet

// Runtime-neutral task-shape plan attrs reach ARTS through CODIR codelets.

// CHECK-LABEL: func.func @main
// CHECK: arts.db_alloc
// CHECK: arts.db_acquire[<out>]
// CHECK: arts.edt <task> <intranode>
// CHECK-SAME: params(
// CHECK: arts.db_ref
// CHECK: memref.store

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<8xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    sde.cu_region <parallel> {
      sde.su_iterate (%c0) to (%c8) step (%c1) {
      ^bb0(%i: index):
        %v = arith.index_cast %i : index to i32
        memref.store %v, %A[%i] : memref<8xi32>
        sde.yield
      } {asyncStrategy = #sde.async_strategy<advance_edt>, repetitionStructure = #sde.repetition_structure<full_timestep>}
      sde.yield
    }
    return
  }
}
