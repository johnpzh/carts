// RUN: not %carts-compile %s --O3 --arts-config %arts_config --start-from openmp-to-arts --pipeline openmp-to-arts 2>&1 | %FileCheck %s

// CHECK: sde.async_strategy cps_chain requires SDE CPS dataflow transformation before Core materialization

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @raw_cps_chain(%A: memref<16xf64>, %B: memref<16xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    arts_sde.cu_region <parallel> scope(<local>) {
      arts_sde.su_iterate (%c0) to (%c16) step (%c1) {
      ^bb0(%i: index):
        %v = memref.load %A[%i] : memref<16xf64>
        memref.store %v, %B[%i] : memref<16xf64>
        arts_sde.yield
      } {asyncStrategy = #arts_sde.async_strategy<cps_chain>,
         cps_group_id = 0 : i64,
         cps_stage_count = 1 : i64,
         cps_stage_index = 0 : i64,
         repetitionStructure = #arts_sde.repetition_structure<full_timestep>}
      arts_sde.yield
    }
    return
  }
}
