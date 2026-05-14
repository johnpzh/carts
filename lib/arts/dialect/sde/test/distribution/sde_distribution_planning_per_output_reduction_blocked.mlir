// RUN: %carts-compile %s --O3 --arts-config %arts_config --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=SDE
// RUN: %carts-compile %s --O3 --arts-config %arts_config --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=ARTS

// Per-output reductions such as matrix-vector rows do not carry OpenMP
// reduction accumulators. SDE still owns the legality proof: the outer output
// index is parallel, the inner loop is the reduction dimension, and blocked
// distribution is safe because each task owns disjoint output elements.

// SDE-LABEL: // -----// IR Dump After PatternAnalysis (sde-pattern-analysis) //----- //
// SDE: func.func @main
// SDE: sde.su_iterate (%c0) to (%c128) step (%c1) classification(<reduction>)
// SDE: pattern = #sde.pattern<reduction>

// SDE-LABEL: // -----// IR Dump After DistributionPlanning (distribution-planning) //----- //
// SDE: func.func @main
// SDE: sde.cu_region <parallel> {
// SDE: sde.su_distribute <blocked> {
// SDE: sde.su_iterate (%c0) to (%c128) step (%c1) classification(<reduction>)
// SDE: iterationTopology = #sde.iteration_topology<owner_strip>
// SDE-SAME: physicalBlockShape = [16]
// SDE-SAME: physicalOwnerDims = [0]

// ARTS-LABEL: // -----// IR Dump After ConvertSdeToArts (convert-sde-to-arts) //----- //
// ARTS: func.func @main
// ARTS: arts.epoch attributes {
// ARTS-SAME: distribution_kind = #arts.distribution_kind<block>
// ARTS: arts.edt <task>
// ARTS: depPattern = #arts.dep_pattern<uniform>
// ARTS-SAME: distribution_kind = #arts.distribution_kind<block>
// ARTS-NOT: sde.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<128x128xf64>, %x: memref<128xf64>, %y: memref<128xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %zero = arith.constant 0.0 : f64
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c0) to (%c128) step (%c1) {
          memref.store %zero, %y[%i] : memref<128xf64>
          scf.for %j = %c0 to %c128 step %c1 {
            %acc = memref.load %y[%i] : memref<128xf64>
            %a = memref.load %A[%i, %j] : memref<128x128xf64>
            %v = memref.load %x[%j] : memref<128xf64>
            %prod = arith.mulf %a, %v : f64
            %next = arith.addf %acc, %prod : f64
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
