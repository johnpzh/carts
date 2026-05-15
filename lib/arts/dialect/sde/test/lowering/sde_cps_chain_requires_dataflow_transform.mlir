// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from sde-planning --pipeline codir-to-arts \
// RUN:   | %FileCheck %s --implicit-check-not=sde. --implicit-check-not=codir.codelet

// CHECK-LABEL: func.func @raw_cps_chain
// CHECK: arts.db_alloc
// CHECK: arts.db_alloc
// CHECK: arts.db_acquire[<in>]
// CHECK: arts.db_acquire[<out>]
// CHECK: arts.edt <task> <intranode>
// CHECK: arts.db_ref
// CHECK: memref.load
// CHECK: memref.store

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @raw_cps_chain(%A: memref<16xf64>, %B: memref<16xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    sde.cu_region <parallel> scope(<local>) {
      sde.su_iterate (%c0) to (%c16) step (%c1) {
      ^bb0(%i: index):
        %v = memref.load %A[%i] : memref<16xf64>
        memref.store %v, %B[%i] : memref<16xf64>
        sde.yield
      } {asyncStrategy = #sde.async_strategy<cps_chain>,
         cps_group_id = 0 : i64,
         cps_stage_count = 1 : i64,
         cps_stage_index = 0 : i64,
         repetitionStructure = #sde.repetition_structure<full_timestep>}
      sde.yield
    }
    return
  }
}
