// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_2t.cfg --start-from sde-planning --pipeline sde-planning --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=SDE

// Verify that OpenMP declare_reduction inference recognizes integer bitwise
// and reductions as `land`, selects the target-neutral SDE reduction strategy,
// and preserves the all-ones identity in the combine path.

// SDE-LABEL: // -----// IR Dump After ReductionStrategy (reduction-strategy) //----- //
// SDE: func.func @main
// SDE: sde.cu_region <parallel> {
// SDE: sde.su_iterate (%c0) to (%c16) step (%c1)
// SDE-SAME: reduction{{\[}}#sde.reduction_kind<and>{{\]}}
// SDE-SAME: reduction_strategy(<tree>)
// SDE-NOT: reduction_strategy(<atomic>)
// SDE: arith.andi
// SDE: arith.andi
// SDE-LABEL: // -----// IR Dump After IterationSpaceDecomposition

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  omp.declare_reduction @and_i32 : i32 init {
  ^bb0(%arg0: i32):
    %c-1_i32 = arith.constant -1 : i32
    omp.yield(%c-1_i32 : i32)
  } combiner {
  ^bb0(%lhs: i32, %rhs: i32):
    %combined = arith.andi %lhs, %rhs : i32
    omp.yield(%combined : i32)
  }

  func.func @main(%A: memref<16xi32>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %acc = memref.alloca() : memref<1xi32>
    %acc_cast = memref.cast %acc : memref<1xi32> to memref<?xi32>
    %c-1_i32 = arith.constant -1 : i32
    memref.store %c-1_i32, %acc[%c0] : memref<1xi32>
    omp.parallel {
      omp.wsloop reduction(@and_i32 %acc_cast -> %prv : memref<?xi32>) {
        omp.loop_nest (%i) : index = (%c0) to (%c16) step (%c1) {
          %val = memref.load %A[%i] : memref<16xi32>
          %old = memref.load %prv[%c0] : memref<?xi32>
          %next = arith.andi %old, %val : i32
          memref.store %next, %prv[%c0] : memref<?xi32>
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }
}
