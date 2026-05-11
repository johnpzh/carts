// RUN: %carts-compile %s --O3 --arts-config %arts_config --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=SDE
// RUN: %carts-compile %s --O3 --arts-config %arts_config --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=ARTS

// Verify that SDE now authors a runtime-neutral neighborhood summary before
// crossing into ARTS IR, and that ConvertSdeToArts materializes the matching
// ARTS contract at the boundary.

// SDE-LABEL: // -----// IR Dump After StructuredSummaries (structured-summaries) //----- //
// SDE: func.func @main
// SDE: arts_sde.su_iterate (%c1) to (%c63) step (%c1) classification(<stencil>) {
// SDE: } {accessMaxOffsets = [1, 1], accessMinOffsets = [-1, -1], ownerDims = [0, 1], spatialDims = [0, 1], writeFootprint = [1, 1]}

// ARTS-LABEL: // -----// IR Dump After ConvertSdeToArts (convert-sde-to-arts) //----- //
// ARTS: func.func @main
// ARTS: arts.edt <parallel> <intranode> route(%{{.*}}) attributes {
// ARTS-SAME: arts.pattern_revision = 1 : i64
// ARTS-SAME: depPattern = #arts.dep_pattern<stencil_tiling_nd>
// ARTS-SAME: distribution_pattern = #arts.distribution_pattern<stencil>
// ARTS-SAME: stencil_max_offsets = [1, 1]
// ARTS-SAME: stencil_min_offsets = [-1, -1]
// ARTS-SAME: stencil_owner_dims = [0, 1]
// ARTS-SAME: stencil_spatial_dims = [0, 1]
// ARTS-SAME: stencil_supported_block_halo
// ARTS-SAME: stencil_write_footprint = [1, 1]
// ARTS: arts.for(%c1) to(%c63) step(%c1) {
// ARTS: } {arts.pattern_revision = 1 : i64
// ARTS-SAME: depPattern = #arts.dep_pattern<stencil_tiling_nd>
// ARTS-SAME: distribution_pattern = #arts.distribution_pattern<stencil>
// ARTS-SAME: stencil_max_offsets = [1, 1]
// ARTS-SAME: stencil_min_offsets = [-1, -1]
// ARTS-SAME: stencil_owner_dims = [0, 1]
// ARTS-SAME: stencil_spatial_dims = [0, 1]
// ARTS-SAME: stencil_supported_block_halo
// ARTS-SAME: stencil_write_footprint = [1, 1]
// ARTS-NOT: arts_sde.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<64x64xf64>, %B: memref<64x64xf64>) {
    %c1 = arith.constant 1 : index
    %c63 = arith.constant 63 : index
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c1) to (%c63) step (%c1) {
          scf.for %j = %c1 to %c63 step %c1 {
            %im1 = arith.subi %i, %c1 : index
            %ip1 = arith.addi %i, %c1 : index
            %jm1 = arith.subi %j, %c1 : index
            %jp1 = arith.addi %j, %c1 : index
            %n = memref.load %A[%im1, %j] : memref<64x64xf64>
            %s = memref.load %A[%ip1, %j] : memref<64x64xf64>
            %w = memref.load %A[%i, %jm1] : memref<64x64xf64>
            %e = memref.load %A[%i, %jp1] : memref<64x64xf64>
            %c = memref.load %A[%i, %j] : memref<64x64xf64>
            %s0 = arith.addf %n, %s : f64
            %s1 = arith.addf %w, %e : f64
            %s2 = arith.addf %s0, %s1 : f64
            %sum = arith.addf %s2, %c : f64
            memref.store %sum, %B[%i, %j] : memref<64x64xf64>
          }
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }
}
