// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_64t.cfg --start-from sde-planning --pipeline sde-to-codir --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Two-dimensional elementwise/in-place loops should expose both independent
// owner dimensions to CODIR instead of launching one coarse strip that leaves
// most workers idle.

// CHECK-LABEL: // -----// IR Dump After DistributionPlanning (distribution-planning) //----- //
// CHECK: func.func @elementwise_inplace_2d_owner_tile
// CHECK: sde.su_iterate
// CHECK-SAME: classification(<elementwise>)
// CHECK: iterationTopology = #sde.iteration_topology<owner_tile>
// CHECK-SAME: logicalWorkerSlice = [64, 64, 16]
// CHECK-SAME: physicalBlockShape = [64, 64, 16]
// CHECK-SAME: physicalOwnerDims = [0, 1]
// CHECK-LABEL: // -----// IR Dump After ConvertSdeToCodir
// CHECK: func.func @elementwise_inplace_2d_owner_tile
// CHECK: scf.for {{.*}} step %c64
// CHECK: scf.for {{.*}} step %c64
// CHECK: codir.codelet

module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">>,
  llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128",
  llvm.target_triple = "aarch64-unknown-linux-gnu"
} {
  func.func @elementwise_inplace_2d_owner_tile(%A: memref<512x512x16xf32>, %B: memref<512x512x16xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c512 = arith.constant 512 : index
    sde.cu_region <parallel> {
      sde.su_iterate (%c0, %c0) to (%c512, %c512) step (%c1, %c1) classification(<elementwise>) {
      ^bb0(%b: index, %c: index):
        scf.for %i = %c0 to %c16 step %c1 {
          %old = memref.load %A[%b, %c, %i] : memref<512x512x16xf32>
          %bias = memref.load %B[%b, %c, %i] : memref<512x512x16xf32>
          %next = arith.addf %old, %bias : f32
          memref.store %next, %A[%b, %c, %i] : memref<512x512x16xf32>
        }
        sde.yield
      }
      sde.yield
    }
    return
  }
}
