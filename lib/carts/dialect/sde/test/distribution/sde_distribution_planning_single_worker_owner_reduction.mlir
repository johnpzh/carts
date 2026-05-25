// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_1t.cfg --start-from sde-planning --pipeline sde-planning --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=SDE

// Single-worker runs still need task-shape planning for owner-indexed reductions.
// Without a logical worker slice, later lowering falls back to the source step
// and creates one EDT per element instead of one full-owner task.

// SDE-LABEL: // -----// IR Dump After DistributionPlanning (distribution-planning) //----- //
// SDE: func.func @single_worker_owner_reduction_keeps_full_slice
// SDE: sde.su_iterate (%c0) to (%c128) step (%c1) classification(<reduction>)
// SDE: } {iterationTopology = #sde.iteration_topology<owner_strip>, logicalWorkerSlice = [128, 30], pattern = #sde.pattern<reduction>{{.*}}}
// SDE-LABEL: // -----// IR Dump After IterationSpaceDecomposition

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @single_worker_owner_reduction_keeps_full_slice(%A: memref<128x128xf32>, %out: memref<128x30xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c30 = arith.constant 30 : index
    %c128 = arith.constant 128 : index
    %zero = arith.constant 0.0 : f32
    sde.cu_region <parallel> {
      sde.su_iterate (%c0) to (%c128) step (%c1) classification(<reduction>) {
      ^bb0(%i: index):
        %sum = scf.for %j = %c0 to %c30 step %c1 iter_args(%acc = %zero) -> (f32) {
          %owner = memref.load %A[%i, %j] : memref<128x128xf32>
          %cross = memref.load %A[%j, %i] : memref<128x128xf32>
          %prod = arith.mulf %owner, %cross : f32
          %next = arith.addf %acc, %prod : f32
          scf.yield %next : f32
        }
        memref.store %sum, %out[%i, %c0] : memref<128x30xf32>
        sde.yield
      } {pattern = #sde.pattern<reduction>}
      sde.yield
    }
    return
  }
}
