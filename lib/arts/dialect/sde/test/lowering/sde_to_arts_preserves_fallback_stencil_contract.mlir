// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from openmp-to-arts --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Verify that ConvertSdeToArts preserves an SDE-owned stencil fallback
// contract even when no transient linalg carrier exists in the loop body.

// CHECK-LABEL: // -----// IR Dump After ConvertSdeToArts (convert-sde-to-arts) //----- //
// CHECK: func.func @main
// CHECK: arts.edt <parallel> <intranode> route(%{{.*}}) attributes {
// CHECK-SAME: arts.pattern_revision = 1 : i64
// CHECK-SAME: depPattern = #arts.dep_pattern<stencil_tiling_nd>
// CHECK-SAME: distribution_pattern = #arts.distribution_pattern<stencil>
// CHECK-SAME: no_verify = #arts.no_verify
// CHECK-SAME: planHaloShape = [1]
// CHECK-SAME: planIterationTopology = #arts.plan_iteration_topology<owner_strip>
// CHECK-SAME: planOwnerDims = [0]
// CHECK-SAME: planPhysicalBlockShape = [2, 16]
// CHECK-SAME: stencil_block_shape = [2]
// CHECK-SAME: stencil_max_offsets = [1]
// CHECK-SAME: stencil_min_offsets = [-1]
// CHECK-SAME: stencil_owner_dims = [0]
// CHECK-SAME: stencil_spatial_dims = [0, 1]
// CHECK-SAME: stencil_supported_block_halo
// CHECK-SAME: stencil_write_footprint = [1]
// CHECK: arts.for(%c1) to(%c15) step(%{{.*}}) {
// CHECK: } {arts.pattern_revision = 1 : i64
// CHECK-SAME: depPattern = #arts.dep_pattern<stencil_tiling_nd>
// CHECK-SAME: distribution_pattern = #arts.distribution_pattern<stencil>
// CHECK-SAME: planHaloShape = [1]
// CHECK-SAME: planIterationTopology = #arts.plan_iteration_topology<owner_strip>
// CHECK-SAME: planOwnerDims = [0]
// CHECK-SAME: planPhysicalBlockShape = [2, 16]
// CHECK-SAME: stencil_block_shape = [2]
// CHECK-SAME: stencil_max_offsets = [1]
// CHECK-SAME: stencil_min_offsets = [-1]
// CHECK-SAME: stencil_owner_dims = [0]
// CHECK-SAME: stencil_spatial_dims = [0, 1]
// CHECK-SAME: stencil_supported_block_halo
// CHECK-SAME: stencil_write_footprint = [1]
// CHECK-NOT: linalg.generic

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<16x16xf64>, %B: memref<16x16xf64>) {
    %c1 = arith.constant 1 : index
    %c15 = arith.constant 15 : index
    arts_sde.cu_region <parallel> scope(<local>) {
      arts_sde.su_iterate (%c1) to (%c15) step (%c1) classification(<stencil>) {
      ^bb0(%i: index):
        scf.for %j = %c1 to %c15 step %c1 {
          %im1 = arith.subi %i, %c1 : index
          %ip1 = arith.addi %i, %c1 : index
          %jm1 = arith.subi %j, %c1 : index
          %jp1 = arith.addi %j, %c1 : index
          %n = memref.load %A[%im1, %j] : memref<16x16xf64>
          %s = memref.load %A[%ip1, %j] : memref<16x16xf64>
          %w = memref.load %A[%i, %jm1] : memref<16x16xf64>
          %e = memref.load %A[%i, %jp1] : memref<16x16xf64>
          %c = memref.load %A[%i, %j] : memref<16x16xf64>
          %s0 = arith.addf %n, %s : f64
          %s1 = arith.addf %w, %e : f64
          %s2 = arith.addf %s0, %s1 : f64
          %sum = arith.addf %s2, %c : f64
          memref.store %sum, %B[%i, %j] : memref<16x16xf64>
        }
        arts_sde.yield
      } {accessMaxOffsets = [1, 1], accessMinOffsets = [-1, -1], depFamily = #arts_sde.dep_family<stencil_tiling_nd>, iterationTopology = #arts_sde.iteration_topology<owner_strip>, ownerDims = [0, 1], physicalBlockShape = [2, 16], physicalHaloShape = [1], physicalOwnerDims = [0], spatialDims = [0, 1], writeFootprint = [1, 1]}
      arts_sde.yield
    }
    return
  }
}
