// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_multinode_64x64.cfg --start-from sde-planning --pipeline sde-to-codir --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=SDE

// Transposed-GEMV style loops reduce a nested dimension into an output slice
// owned by the scheduling-unit IV. SDE should expose every owner slice when the
// owner domain is smaller than the high-node worker contract; keeping the tile
// at one avoids an artificial minimum-grain floor that would serialize
// otherwise independent output reductions.

// SDE-LABEL: // -----// IR Dump After Tiling (tiling) //----- //
// SDE: func.func @main
// SDE: sde.su_distribute <blocked> {
// SDE: sde.su_iterate (%c0) to (%c128) step (%c1) reduction_strategy(<local_accumulate>) classification(<elementwise_pipeline>)
// SDE: iterationTopology = #sde.iteration_topology<owner_strip>
// SDE-SAME: logicalWorkerSlice = [1]
// SDE-SAME: physicalBlockShape = [1]
// SDE-SAME: physicalOwnerDims = [0]
// SDE-LABEL: // -----// IR Dump After ConvertSdeToCodir (convert-sde-to-codir) //----- //
// SDE: func.func @main
// SDE: codir.codelet
// SDE-SAME: logical_worker_slice = [1]
// SDE-SAME: pattern = #codir.pattern<elementwise_pipeline>
// SDE-SAME: tile_shape = [1]

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<128x128xf64>, %r: memref<128xf64>, %s: memref<128xf64>) {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f64
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%j) : index = (%c0) to (%c128) step (%c1) {
          memref.store %zero, %s[%j] : memref<128xf64>
          scf.for %i = %c0 to %c128 step %c1 {
            %old = memref.load %s[%j] : memref<128xf64>
            %rv = memref.load %r[%i] : memref<128xf64>
            %av = memref.load %A[%i, %j] : memref<128x128xf64>
            %prod = arith.mulf %rv, %av : f64
            %next = arith.addf %old, %prod : f64
            memref.store %next, %s[%j] : memref<128xf64>
          }
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }
}
