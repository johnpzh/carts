// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from sde-planning \
// RUN:   --pipeline sde-to-codir --mlir-print-ir-after-all 2>&1 | %FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<128x128xf64>, %y: memref<128xf64>) {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f64
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c0) to (%c128) step (%c1) {
          memref.store %zero, %y[%i] : memref<128xf64>
          scf.for %j = %c0 to %c128 step %c1 {
            %old = memref.load %y[%i] : memref<128xf64>
            %a = memref.load %A[%i, %j] : memref<128x128xf64>
            %next = arith.addf %old, %a : f64
            memref.store %next, %y[%i] : memref<128xf64>
          }
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }
}

// CHECK-LABEL: // -----// IR Dump After PatternAnalysis (sde-pattern-analysis) //----- //
// CHECK: sde.su_iterate
// CHECK-SAME: classification(<elementwise_pipeline>)
// CHECK: partialReduction
// CHECK-SAME: partialReductionDims = [1]
// CHECK-SAME: partialReductionOwnerDims = [0]
// CHECK-LABEL: // -----// IR Dump After ReductionStrategy (reduction-strategy) //----- //
// CHECK: reduction_strategy(<local_accumulate>)
// CHECK-LABEL: // -----// IR Dump After ReductionPlanning (reduction-planning) //----- //
// CHECK: codir.codelet
// CHECK-SAME: partial_reduction
// CHECK-SAME: partial_reduction_dep_result_dim_maps = {{\[\[}}0], [0, -1]]

