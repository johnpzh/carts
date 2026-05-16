// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_2t.cfg --start-from sde-planning --pipeline sde-planning --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Verify that loop-uniform multi-reduction selection still prioritizes
// local_accumulate when the reduction loop already contains nested sequential
// work.

// CHECK-LABEL: // -----// IR Dump After ReductionStrategy (reduction-strategy) //----- //
// CHECK: func.func @main
// CHECK: sde.cu_region <parallel> {
// CHECK: sde.su_iterate (%c0) to (%c16) step (%c1)
// CHECK-SAME: #sde.reduction_kind<add>, #sde.reduction_kind<mul>
// CHECK-SAME: reduction_strategy(<local_accumulate>)
// CHECK-NOT: arts.for
// CHECK: // -----// IR Dump After DistributionPlanning (distribution-planning) //----- //

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<16x4xi32>, %B: memref<16x4xi32>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %sum = memref.alloca() : memref<1xi32>
    %prod = memref.alloca() : memref<1xi32>
    %sum_cast = memref.cast %sum : memref<1xi32> to memref<?xi32>
    %prod_cast = memref.cast %prod : memref<1xi32> to memref<?xi32>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    memref.store %c0_i32, %sum[%c0] : memref<1xi32>
    memref.store %c1_i32, %prod[%c0] : memref<1xi32>
    sde.cu_region <parallel> {
      "sde.su_iterate"(%c0, %c16, %c1, %sum_cast, %prod_cast) ({
      ^bb0(%iv: index):
        scf.for %j = %c0 to %c4 step %c1 {
          %lhs = memref.load %A[%iv, %j] : memref<16x4xi32>
          %acc0 = memref.load %sum_cast[%c0] : memref<?xi32>
          %next0 = arith.addi %acc0, %lhs : i32
          memref.store %next0, %sum_cast[%c0] : memref<?xi32>
          %rhs = memref.load %B[%iv, %j] : memref<16x4xi32>
          %acc1 = memref.load %prod_cast[%c0] : memref<?xi32>
          %next1 = arith.muli %acc1, %rhs : i32
          memref.store %next1, %prod_cast[%c0] : memref<?xi32>
        }
        "sde.yield"() : () -> ()
      }) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 2>, reductionKinds = [#sde.reduction_kind<add>, #sde.reduction_kind<mul>]} : (index, index, index, memref<?xi32>, memref<?xi32>) -> ()
      sde.yield
    }
    return
  }
}
