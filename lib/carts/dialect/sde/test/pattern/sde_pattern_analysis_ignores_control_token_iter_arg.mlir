// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from sde-planning --pipeline sde-planning --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Rank-0 i1 tensor iter_args are scalar control carriers introduced by scalar
// threading. They must not turn an otherwise elementwise scheduling unit into
// a reduction or a distributable result-producing task. The tensor carrier is
// rejected at the CODIR-to-ARTS boundary until CODIR owns explicit control deps.

// CHECK-LABEL: // -----// IR Dump After PatternAnalysis (sde-pattern-analysis) //----- //
// CHECK: func.func @control_token_elementwise
// CHECK: sde.su_iterate (%c0) to (%c128) step (%c1) classification(<elementwise>) iter_args
// CHECK: pattern = #sde.pattern<uniform>
// CHECK-LABEL: // -----// IR Dump After DistributionPlanning (distribution-planning) //----- //
// CHECK: func.func @control_token_elementwise
// CHECK: sde.su_iterate (%c0) to (%c128) step (%c1) classification(<elementwise>) iter_args
// CHECK: } {iterationTopology = #sde.iteration_topology<owner_strip>
// CHECK-SAME: physicalOwnerDims = [0]

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @control_token_elementwise(%A: memref<128xf64>, %B: memref<128xf64>, %flag: tensor<i1>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %true = arith.constant true
    %result = sde.cu_region <parallel> iter_args(%arg0 = %flag : tensor<i1>) -> (tensor<i1>) {
      %iter = sde.su_iterate (%c0) to (%c128) step (%c1) iter_args(%arg1 = %arg0 : tensor<i1>) -> (tensor<i1>) {
      ^bb0(%i: index, %arg2: tensor<i1>):
        %inner = sde.cu_region <parallel> iter_args(%arg3 = %arg2 : tensor<i1>) -> (tensor<i1>) {
          %v = memref.load %A[%i] : memref<128xf64>
          memref.store %v, %B[%i] : memref<128xf64>
          %next = tensor.insert %true into %arg3[] : tensor<i1>
          sde.yield %next : tensor<i1>
        }
        sde.yield %inner : tensor<i1>
      }
      sde.yield %iter : tensor<i1>
    }
    return
  }
}
