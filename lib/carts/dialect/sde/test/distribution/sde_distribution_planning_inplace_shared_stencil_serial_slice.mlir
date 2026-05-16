// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from sde-planning --pipeline codir-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// In-place shared-state stencils cannot be split into independent owner tiles
// until SDE materializes a wavefront/token dataflow plan. They can still reduce
// launch overhead by strip-mining consecutive owner rows into a serial chunk
// while preserving the coarse readwrite dependency.

// CHECK-LABEL: // -----// IR Dump After DistributionPlanning (distribution-planning) //----- //
// CHECK: func.func @inplace_shared_stencil_serial_slice
// CHECK: %[[C8:.*]] = arith.constant 8 : index
// CHECK: %[[HALO:.*]] = arith.maxui %[[C8]], {{.*}} : index
// CHECK: %[[BOUNDED:.*]] = arith.minui %[[HALO]], {{.*}} : index
// CHECK: %[[STEP:.*]] = arith.muli %c1, %[[BOUNDED]] : index
// CHECK: sde.su_iterate (%c1) to (%c65) step (%[[STEP]]) classification(<stencil>) {
// CHECK: scf.for {{.*}} = %arg{{[0-9]+}} to {{.*}} step %c1 {
// CHECK: } {
// CHECK-SAME: inPlaceSharedState
// CHECK-SAME: ownerDims = [0, 1]
// CHECK-SAME: pattern = #sde.pattern<stencil_tiling_nd>
// CHECK-LABEL: // -----// IR Dump After ConvertSdeToCodir (convert-sde-to-codir) //----- //
// CHECK: scf.for {{.*}} = %c1 to %c65 step %[[CODIR_STEP:.*]] {
// CHECK: codir.codelet
// CHECK-SAME: dep_modes = [#codir.access_mode<readwrite>]
// CHECK-SAME: in_place_shared_state
// CHECK-SAME: pattern = #codir.pattern<stencil_tiling_nd>
// CHECK-LABEL: // -----// IR Dump After ConvertCodirToArts (convert-codir-to-arts) //----- //
// CHECK: scf.for {{.*}} = %c1{{.*}} to %c65 step %[[ARTS_STEP:.*]] {
// CHECK: arts.edt <task>
// CHECK-SAME: depPattern = #arts.dep_pattern<stencil_tiling_nd>
// CHECK-SAME: distribution_pattern = #arts.distribution_pattern<stencil>
// CHECK-SAME: inPlaceSharedState
// CHECK-NOT: sde.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @inplace_shared_stencil_serial_slice(%A: memref<66x66xf64>) {
    %c1 = arith.constant 1 : index
    %c65 = arith.constant 65 : index
    %cst = arith.constant 9.000000e+00 : f64
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c1) to (%c65) step (%c1) {
          scf.for %j = %c1 to %c65 step %c1 {
            %im1 = arith.subi %i, %c1 : index
            %ip1 = arith.addi %i, %c1 : index
            %jm1 = arith.subi %j, %c1 : index
            %jp1 = arith.addi %j, %c1 : index
            %v0 = memref.load %A[%im1, %jm1] : memref<66x66xf64>
            %v1 = memref.load %A[%im1, %j] : memref<66x66xf64>
            %s0 = arith.addf %v0, %v1 : f64
            %v2 = memref.load %A[%im1, %jp1] : memref<66x66xf64>
            %s1 = arith.addf %s0, %v2 : f64
            %v3 = memref.load %A[%i, %jm1] : memref<66x66xf64>
            %s2 = arith.addf %s1, %v3 : f64
            %v4 = memref.load %A[%i, %j] : memref<66x66xf64>
            %s3 = arith.addf %s2, %v4 : f64
            %v5 = memref.load %A[%i, %jp1] : memref<66x66xf64>
            %s4 = arith.addf %s3, %v5 : f64
            %v6 = memref.load %A[%ip1, %jm1] : memref<66x66xf64>
            %s5 = arith.addf %s4, %v6 : f64
            %v7 = memref.load %A[%ip1, %j] : memref<66x66xf64>
            %s6 = arith.addf %s5, %v7 : f64
            %v8 = memref.load %A[%ip1, %jp1] : memref<66x66xf64>
            %s7 = arith.addf %s6, %v8 : f64
            %avg = arith.divf %s7, %cst : f64
            memref.store %avg, %A[%i, %j] : memref<66x66xf64>
          }
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }
}
