// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_multinode_8x64.cfg --start-from sde-planning --pipeline sde-to-codir --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=MN8

// Owner-local pipeline reductions need more than one task wave when the
// compile-time target is multinode.  With 8 nodes x 64 workers, the old
// one-wave policy used 8192 / 512 = 16 rows per task.  The cost-model wave
// policy keeps the same amortization floor but exposes more independent work.

// MN8-LABEL: // -----// IR Dump After Tiling (tiling) //----- //
// MN8: func.func @transpose_gemv_owner_pipeline
// MN8: %[[STEP:.*]] = arith.muli %c1, %c10{{(_[0-9]+)?}} : index
// MN8: sde.su_iterate (%c0) to (%c8192) step (%[[STEP]]) classification(<elementwise_pipeline>)
// MN8: physicalBlockShape = [10]
// MN8-LABEL: // -----// IR Dump After ConvertSdeToCodir (convert-sde-to-codir) //----- //
// MN8: func.func @transpose_gemv_owner_pipeline
// MN8: codir.codelet
// MN8-SAME: logical_worker_slice = [10]
// MN8-SAME: pattern = #codir.pattern<elementwise_pipeline>
// MN8-SAME: tile_shape = [10]

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @transpose_gemv_owner_pipeline(%A: memref<8192x8192xf64>, %r: memref<8192xf64>, %s: memref<8192xf64>) {
    %c0 = arith.constant 0 : index
    %c8192 = arith.constant 8192 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f64
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%j) : index = (%c0) to (%c8192) step (%c1) {
          memref.store %zero, %s[%j] : memref<8192xf64>
          scf.for %i = %c0 to %c8192 step %c1 {
            %old = memref.load %s[%j] : memref<8192xf64>
            %rv = memref.load %r[%i] : memref<8192xf64>
            %av = memref.load %A[%i, %j] : memref<8192x8192xf64>
            %prod = arith.mulf %rv, %av : f64
            %next = arith.addf %old, %prod : f64
            memref.store %next, %s[%j] : memref<8192xf64>
          }
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }
}
