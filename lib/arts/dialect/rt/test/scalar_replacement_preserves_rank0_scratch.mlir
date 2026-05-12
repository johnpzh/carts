// RUN: %carts-compile %s --arts-config %arts_config --start-from pre-lowering --pipeline pre-lowering | %FileCheck %s

// Rank-0 private scratch accumulators are intentionally left in memory form.
// LLVM/SROA handles these better than forcing an RT-level loop-carried scalar
// in large Monte Carlo workers.

// CHECK-LABEL: func.func @rank0_scratch_stays_memref
// CHECK: %[[SCRATCH:.+]] = memref.alloca() : memref<f64>
// CHECK: scf.for
// CHECK: memref.load %[[SCRATCH]][] : memref<f64>
// CHECK: memref.store {{.*}}, %[[SCRATCH]][] : memref<f64>
// CHECK-NOT: iter_args

module attributes {arts.runtime_total_nodes = 1 : i64, arts.runtime_total_workers = 8 : i64, dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<[8, 32]> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i16, dense<[16, 32]> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @rank0_scratch_stays_memref() -> f64 {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.000000e+00 : f64
    %one = arith.constant 1.000000e+00 : f64
    %scratch = memref.alloca() : memref<f64>
    memref.store %zero, %scratch[] : memref<f64>
    scf.for %i = %c0 to %c16 step %c1 {
      %old = memref.load %scratch[] : memref<f64>
      %new = arith.addf %old, %one : f64
      memref.store %new, %scratch[] : memref<f64>
    }
    %result = memref.load %scratch[] : memref<f64>
    return %result : f64
  }
}
