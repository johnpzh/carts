// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_8t.cfg --start-from sde-planning --pipeline codir-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Tiling owns an early ND physical stencil plan when it can prove an
// out-of-place structured stencil. The plan must map semantic SDE owner loop
// dims through the output layout instead of treating loop dims as physical
// memref dims. Here the store is B[j, i], so semantic owner dims [0, 1] map to
// physical owner dims [1, 0].

// CHECK-LABEL: // -----// IR Dump After Tiling (tiling) //----- //
// CHECK: func.func @transposed_out_of_place_stencil
// CHECK: sde.su_iterate
// CHECK-SAME: classification(<stencil>)
// CHECK: } {accessMaxOffsets
// CHECK-SAME: iterationTopology = #sde.iteration_topology<owner_tile>
// CHECK-SAME: physicalBlockShape = [64, 32]
// CHECK-SAME: physicalHaloShape = [1, 1]
// CHECK-SAME: physicalOwnerDims = [1, 0]
// CHECK-LABEL: // -----// IR Dump After Vectorization

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @transposed_out_of_place_stencil(%A: memref<256x64xf64>, %B: memref<256x64xf64>) {
    %c1 = arith.constant 1 : index
    %c63 = arith.constant 63 : index
    %c255 = arith.constant 255 : index
    sde.cu_region <parallel> {
      sde.su_iterate (%c1, %c1) to (%c63, %c255) step (%c1, %c1) {
      ^bb0(%i: index, %j: index):
        %im1 = arith.subi %i, %c1 : index
        %ip1 = arith.addi %i, %c1 : index
        %jm1 = arith.subi %j, %c1 : index
        %jp1 = arith.addi %j, %c1 : index
        %x0 = memref.load %A[%j, %im1] : memref<256x64xf64>
        %x1 = memref.load %A[%j, %ip1] : memref<256x64xf64>
        %y0 = memref.load %A[%jm1, %i] : memref<256x64xf64>
        %y1 = memref.load %A[%jp1, %i] : memref<256x64xf64>
        %c = memref.load %A[%j, %i] : memref<256x64xf64>
        %s0 = arith.addf %x0, %x1 : f64
        %s1 = arith.addf %y0, %y1 : f64
        %s2 = arith.addf %s0, %s1 : f64
        %sum = arith.addf %s2, %c : f64
        memref.store %sum, %B[%j, %i] : memref<256x64xf64>
        sde.yield
      }
      sde.yield
    }
    return
  }
}
