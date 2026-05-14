// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from openmp-to-arts --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Verify that an explicit inner local scope remains authoritative even when
// the enclosing parallel region carries an explicit distributed scope.

// CHECK-LABEL: // -----// IR Dump After ScopeSelection (scope-selection) //----- //
// CHECK: func.func @main
// CHECK: sde.cu_region <parallel> scope(<distributed>) {
// CHECK: sde.cu_region <parallel> scope(<local>) {
// CHECK: sde.su_iterate (%c0) to (%c16) step (%c1)

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %tmp = memref.alloca() : memref<16xi32>
    sde.cu_region <parallel> scope(<distributed>) {
      sde.cu_region <parallel> scope(<local>) {
        sde.su_iterate (%c0) to (%c16) step (%c1) {
        ^bb0(%iv: index):
          memref.store %c0_i32, %tmp[%iv] : memref<16xi32>
          sde.yield
        }
        sde.yield
      }
      sde.yield
    }
    return
  }
}
