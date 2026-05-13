// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_multinode.cfg --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=SDE
// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_multinode.cfg --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=ARTS
// RUN: %carts-compile %s --O3 --arts-config %arts_config --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=LOCAL

// Verify that SDE authors an owner-compute distribution advisory for a
// distributed stencil loop, and that ConvertSdeToArts materializes the
// advisory as an ARTS distribution kind at the lowering boundary.
// In-place neighbor stencils keep the original dependency ordering and do not
// receive a physical block DB plan because the update reads neighboring values
// from the same backing store it writes.

// SDE-LABEL: // -----// IR Dump After DistributionPlanning (distribution-planning) //----- //
// SDE: func.func @main
// SDE: arts_sde.cu_region <parallel> scope(<distributed>) {
// SDE: arts_sde.su_distribute <owner_compute> {
// SDE: arts_sde.su_iterate (%c1) to (%c63) step (%c1) classification(<stencil>) {
// SDE: } {accessMaxOffsets
// SDE-SAME: depFamily = #arts_sde.dep_family<stencil_tiling_nd>
// SDE-SAME: iterationTopology = #arts_sde.iteration_topology<owner_strip>
// SDE-SAME: logicalWorkerSlice = [2, 64]
// SDE-SAME: physicalBlockShape = [2, 64]
// SDE-SAME: physicalHaloShape = [1]
// SDE-SAME: physicalOwnerDims = [0]
// SDE-NOT: {{plan[A-Z]}}
// SDE-LABEL: // -----// IR Dump After IterationSpaceDecomposition

// ARTS-LABEL: // -----// IR Dump After ConvertSdeToArts (convert-sde-to-arts) //----- //
// ARTS: func.func @main
// ARTS: arts.edt <parallel> <internode> route(%{{.*}}) attributes {
// ARTS-SAME: {{.*}}depPattern = #arts.dep_pattern<stencil_tiling_nd>{{.*}}distribution_pattern = #arts.distribution_pattern<stencil>
// ARTS-SAME: {{.*}}planHaloShape = [1]
// ARTS-SAME: {{.*}}planLogicalWorkerSlice = [2, 64]
// ARTS-SAME: {{.*}}planOwnerDims = [0]
// ARTS-SAME: {{.*}}planPhysicalBlockShape = [2, 64]
// ARTS: arts.for(%c1) to(%c63) step(%c1) {
// ARTS: } {arts.pattern_revision = 1 : i64
// ARTS-SAME: {{.*}}depPattern = #arts.dep_pattern<stencil_tiling_nd>{{.*}}distribution_pattern = #arts.distribution_pattern<stencil>
// ARTS-SAME: {{.*}}planHaloShape = [1]
// ARTS-SAME: {{.*}}planLogicalWorkerSlice = [2, 64]
// ARTS-SAME: {{.*}}planOwnerDims = [0]
// ARTS-SAME: {{.*}}planPhysicalBlockShape = [2, 64]
// ARTS-NOT: arts_sde.

// LOCAL-LABEL: // -----// IR Dump After DistributionPlanning (distribution-planning) //----- //
// LOCAL: func.func @inplace_serial_stencil
// LOCAL: arts_sde.su_iterate
// LOCAL-SAME: classification(<stencil>)
// LOCAL: } {accessMaxOffsets
// LOCAL-SAME: inPlaceSharedState
// LOCAL-NOT: physicalBlockShape
// LOCAL-LABEL: // -----// IR Dump After IterationSpaceDecomposition

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

  func.func @inplace_serial_stencil(%A: memref<32x32xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c31 = arith.constant 31 : index
    %cst = arith.constant 9.000000e+00 : f64
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c1) to (%c31) step (%c1) {
          scf.for %j = %c1 to %c31 step %c1 {
            %im1 = arith.subi %i, %c1 : index
            %ip1 = arith.addi %i, %c1 : index
            %jm1 = arith.subi %j, %c1 : index
            %jp1 = arith.addi %j, %c1 : index
            %a0 = memref.load %A[%im1, %jm1] : memref<32x32xf64>
            %a1 = memref.load %A[%im1, %j] : memref<32x32xf64>
            %s0 = arith.addf %a0, %a1 : f64
            %a2 = memref.load %A[%im1, %jp1] : memref<32x32xf64>
            %s1 = arith.addf %s0, %a2 : f64
            %a3 = memref.load %A[%i, %jm1] : memref<32x32xf64>
            %s2 = arith.addf %s1, %a3 : f64
            %a4 = memref.load %A[%i, %j] : memref<32x32xf64>
            %s3 = arith.addf %s2, %a4 : f64
            %a5 = memref.load %A[%i, %jp1] : memref<32x32xf64>
            %s4 = arith.addf %s3, %a5 : f64
            %a6 = memref.load %A[%ip1, %jm1] : memref<32x32xf64>
            %s5 = arith.addf %s4, %a6 : f64
            %a7 = memref.load %A[%ip1, %j] : memref<32x32xf64>
            %s6 = arith.addf %s5, %a7 : f64
            %a8 = memref.load %A[%ip1, %jp1] : memref<32x32xf64>
            %s7 = arith.addf %s6, %a8 : f64
            %avg = arith.divf %s7, %cst : f64
            memref.store %avg, %A[%i, %j] : memref<32x32xf64>
          }
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }
}
