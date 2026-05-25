// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_multinode.cfg --start-from sde-planning --pipeline codir-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Higher-order stencils should not use extent-only worker factoring: splitting
// the wider halo dimension first inflates every owner tile's expanded stencil
// footprint. The two-node test config exposes 32 target stencil tasks; with
// shape 64x64 and halo radii [4, 1], the halo-costed grid is 4x8, yielding
// physical blocks [16, 8] instead of the old extent-only [8, 16].

// CHECK-LABEL: // -----// IR Dump After DistributionPlanning (distribution-planning) //----- //
// CHECK: func.func @main
// CHECK: sde.su_iterate (%c4, %c1) to (%c60, %c63) step
// CHECK: } {accessMaxOffsets
// CHECK-SAME: iterationTopology = #sde.iteration_topology<owner_tile>
// CHECK-SAME: ownerDims = [0, 1]
// CHECK-SAME: pattern = #sde.pattern<higher_order_stencil>
// CHECK-SAME: physicalBlockShape = [16, 8]
// CHECK-SAME: physicalHaloShape = [4, 1]
// CHECK-SAME: physicalOwnerDims = [0, 1]
// CHECK-LABEL: // -----// IR Dump After IterationSpaceDecomposition

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<64x64xf64>, %B: memref<64x64xf64>) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c60 = arith.constant 60 : index
    %c63 = arith.constant 63 : index
    sde.cu_region <parallel> {
      sde.su_iterate (%c4, %c1) to (%c60, %c63) step (%c1, %c1) classification(<stencil>) {
      ^bb0(%i: index, %j: index):
        %im4 = arith.subi %i, %c4 : index
        %ip4 = arith.addi %i, %c4 : index
        %jm1 = arith.subi %j, %c1 : index
        %jp1 = arith.addi %j, %c1 : index
        %x0 = memref.load %A[%im4, %j] : memref<64x64xf64>
        %x1 = memref.load %A[%ip4, %j] : memref<64x64xf64>
        %y0 = memref.load %A[%i, %jm1] : memref<64x64xf64>
        %y1 = memref.load %A[%i, %jp1] : memref<64x64xf64>
        %c = memref.load %A[%i, %j] : memref<64x64xf64>
        %s0 = arith.addf %x0, %x1 : f64
        %s1 = arith.addf %y0, %y1 : f64
        %s2 = arith.addf %s0, %s1 : f64
        %sum = arith.addf %s2, %c : f64
        memref.store %sum, %B[%i, %j] : memref<64x64xf64>
        sde.yield
      } {accessMaxOffsets = [4, 1], accessMinOffsets = [-4, -1], ownerDims = [0, 1], spatialDims = [0, 1], writeFootprint = [1, 1]}
      sde.yield
    }
    return
  }
}
