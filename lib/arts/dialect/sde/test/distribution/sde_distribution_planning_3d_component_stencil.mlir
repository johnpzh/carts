// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_multinode.cfg --start-from openmp-to-arts --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=SDE
// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_multinode.cfg --start-from openmp-to-arts --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=ARTS

// SDE owns the physical shape for component stencils: distribute the three
// spatial dimensions and leave the component dimension local in the DB block.

// SDE-LABEL: // -----// IR Dump After DistributionPlanning (distribution-planning) //----- //
// SDE: func.func @main
// SDE: arts_sde.cu_region <parallel> scope(<distributed>) {
// SDE: arts_sde.su_distribute <owner_compute> {
// SDE: arts_sde.su_iterate (%c1, %c1, %c1, %c0) to (%c15, %c15, %c15, %c3) step (%c1, %c1, %c1, %c1) classification(<stencil>) {
// SDE: } {accessMaxOffsets
// SDE-SAME: depFamily = #arts_sde.dep_family<cross_dim_stencil_3d>
// SDE-SAME: iterationTopology = #arts_sde.iteration_topology<owner_tile>
// SDE-SAME: logicalWorkerSlice = [4, 8, 8, 3]
// SDE-SAME: physicalBlockShape = [4, 8, 8, 3]
// SDE-SAME: physicalHaloShape = [1, 1, 1]
// SDE-SAME: physicalOwnerDims = [0, 1, 2]
// SDE-NOT: {{plan[A-Z]}}
// SDE-LABEL: // -----// IR Dump After IterationSpaceDecomposition

// ARTS-LABEL: // -----// IR Dump After ConvertSdeToArts (convert-sde-to-arts) //----- //
// ARTS: func.func @main
// ARTS: arts.edt <parallel> <internode> route(%{{.*}}) attributes {
// ARTS-SAME: depPattern = #arts.dep_pattern<cross_dim_stencil_3d>
// ARTS-SAME: distribution_kind = #arts.distribution_kind<block>
// ARTS-SAME: planHaloShape = [1, 1, 1]
// ARTS-SAME: planIterationTopology = #arts.plan_iteration_topology<owner_tile>
// ARTS-SAME: planLogicalWorkerSlice = [4, 8, 8, 3]
// ARTS-SAME: planOwnerDims = [0, 1, 2]
// ARTS-SAME: planPhysicalBlockShape = [4, 8, 8, 3]
// ARTS-SAME: stencil_block_shape = [4, 8, 8]
// ARTS-SAME: stencil_owner_dims = [0, 1, 2]
// ARTS: arts.for(%c1, %c1, %c1, %c0) to(%c15, %c15, %c15, %c3) step(%c1, %c1, %c1, %c1) {
// ARTS: } {arts.pattern_revision = 1 : i64
// ARTS-SAME: planLogicalWorkerSlice = [4, 8, 8, 3]
// ARTS-SAME: planOwnerDims = [0, 1, 2]
// ARTS-SAME: planPhysicalBlockShape = [4, 8, 8, 3]
// ARTS-NOT: arts_sde.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<16x16x16x3xf64>, %B: memref<16x16x16x3xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c15 = arith.constant 15 : index
    arts_sde.cu_region <parallel> scope(<distributed>) {
      arts_sde.su_iterate (%c1, %c1, %c1, %c0) to (%c15, %c15, %c15, %c3) step (%c1, %c1, %c1, %c1) {
      ^bb0(%i: index, %j: index, %k: index, %c: index):
        %im1 = arith.subi %i, %c1 : index
        %ip1 = arith.addi %i, %c1 : index
        %jm1 = arith.subi %j, %c1 : index
        %jp1 = arith.addi %j, %c1 : index
        %km1 = arith.subi %k, %c1 : index
        %kp1 = arith.addi %k, %c1 : index
        %x0 = memref.load %A[%im1, %j, %k, %c] : memref<16x16x16x3xf64>
        %x1 = memref.load %A[%ip1, %j, %k, %c] : memref<16x16x16x3xf64>
        %y0 = memref.load %A[%i, %jm1, %k, %c] : memref<16x16x16x3xf64>
        %y1 = memref.load %A[%i, %jp1, %k, %c] : memref<16x16x16x3xf64>
        %z0 = memref.load %A[%i, %j, %km1, %c] : memref<16x16x16x3xf64>
        %z1 = memref.load %A[%i, %j, %kp1, %c] : memref<16x16x16x3xf64>
        %s0 = arith.addf %x0, %x1 : f64
        %s1 = arith.addf %y0, %y1 : f64
        %s2 = arith.addf %z0, %z1 : f64
        %s3 = arith.addf %s0, %s1 : f64
        %sum = arith.addf %s3, %s2 : f64
        memref.store %sum, %B[%i, %j, %k, %c] : memref<16x16x16x3xf64>
        arts_sde.yield
      }
      arts_sde.yield
    }
    return
  }
}
