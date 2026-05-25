// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from sde-planning --pipeline sde-planning --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// A loop can be classified as reduction even without OpenMP reduction
// accumulator operands when its write map does not cover the loop IV. That is
// still a true memory reduction and must not be tiled into parallel owner
// chunks that race on the same scalar output.

// CHECK-LABEL: // -----// IR Dump After PatternAnalysis (sde-pattern-analysis) //----- //
// CHECK: func.func @zero_accumulator_memory_reduction
// CHECK: sde.su_iterate (%c0) to (%c128) step (%c1) classification(<reduction>) {
// CHECK-LABEL: // -----// IR Dump After Tiling (tiling) //----- //
// CHECK: func.func @zero_accumulator_memory_reduction
// CHECK: sde.su_iterate (%c0) to (%c128) step (%c1) classification(<reduction>) {
// CHECK-NOT: arith.muli %c1

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @zero_accumulator_memory_reduction(%A: memref<128xf64>, %out: memref<1xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    sde.cu_region <parallel> {
      sde.su_iterate (%c0) to (%c128) step (%c1) {
      ^bb0(%i: index):
        %v = memref.load %A[%i] : memref<128xf64>
        %old = memref.load %out[%c0] : memref<1xf64>
        %sum = arith.addf %old, %v : f64
        memref.store %sum, %out[%c0] : memref<1xf64>
        sde.yield
      }
      sde.yield
    }
    return
  }
}
