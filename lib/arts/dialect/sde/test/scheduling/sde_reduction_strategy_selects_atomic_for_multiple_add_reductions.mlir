// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_64t.cfg --start-from openmp-to-arts --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Verify that loop-uniform multi-reduction selection still takes the atomic
// path when every preserved reduction kind is atomic-capable.

// CHECK-LABEL: // -----// IR Dump After ReductionStrategy (reduction-strategy) //----- //
// CHECK: func.func @main
// CHECK: sde.cu_region <parallel> scope(<local>) {
// CHECK: sde.su_iterate (%c0) to (%c128) step (%c1)
// CHECK-SAME: #sde.reduction_kind<add>, #sde.reduction_kind<add>
// CHECK-SAME: reduction_strategy(<atomic>)
// CHECK-NOT: arts.for
// CHECK: // -----// IR Dump After DistributionPlanning (distribution-planning) //----- //

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<128xi32>, %B: memref<128xi32>) {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %sum0 = memref.alloca() : memref<1xi32>
    %sum1 = memref.alloca() : memref<1xi32>
    %sum0_cast = memref.cast %sum0 : memref<1xi32> to memref<?xi32>
    %sum1_cast = memref.cast %sum1 : memref<1xi32> to memref<?xi32>
    %c0_i32 = arith.constant 0 : i32
    memref.store %c0_i32, %sum0[%c0] : memref<1xi32>
    memref.store %c0_i32, %sum1[%c0] : memref<1xi32>
    sde.cu_region <parallel> scope(<local>) {
      "sde.su_iterate"(%c0, %c128, %c1, %sum0_cast, %sum1_cast) ({
      ^bb0(%iv: index):
        %lhs = memref.load %A[%iv] : memref<128xi32>
        %acc0 = memref.load %sum0_cast[%c0] : memref<?xi32>
        %next0 = arith.addi %acc0, %lhs : i32
        memref.store %next0, %sum0_cast[%c0] : memref<?xi32>
        %rhs = memref.load %B[%iv] : memref<128xi32>
        %acc1 = memref.load %sum1_cast[%c0] : memref<?xi32>
        %next1 = arith.addi %acc1, %rhs : i32
        memref.store %next1, %sum1_cast[%c0] : memref<?xi32>
        "sde.yield"() : () -> ()
      }) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 2>, reductionKinds = [#sde.reduction_kind<add>, #sde.reduction_kind<add>]} : (index, index, index, memref<?xi32>, memref<?xi32>) -> ()
      sde.yield
    }
    return
  }
}
