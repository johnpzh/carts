// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_64t.cfg \
// RUN:   --start-from sde-planning --pipeline sde-to-codir --mlir-print-ir-after-all 2>&1 \
// RUN:   | %FileCheck %s

// Tiling can rewrite one scheduling-unit iteration into a small strip-mined
// chunk. Existing physical block metadata must still describe the memory span
// owned by one future EDT/DB block, so SDE rounds the block shape up to the
// tiled step instead of leaving a block-local view smaller than the codelet.

// CHECK-LABEL: // -----// IR Dump After Tiling (tiling) //----- //
// CHECK: func.func @owner_slice_write_only
// CHECK: %[[STEP:.*]] = arith.muli %c1, %c3{{(_[0-9]+)?}} : index
// CHECK: sde.su_iterate (%c0) to (%c128) step (%[[STEP]]) classification(<elementwise>) {
// CHECK: } {iterationTopology = #sde.iteration_topology<owner_strip>
// CHECK-SAME: logicalWorkerSlice = [18, 64]
// CHECK-SAME: physicalBlockShape = [18, 64]
// CHECK-LABEL: // -----// IR Dump After ConvertSdeToCodir (convert-sde-to-codir) //----- //
// CHECK: func.func @owner_slice_write_only
// CHECK: scf.for %{{.*}} = %c0 to %c128 step %c18
// CHECK: codir.codelet
// CHECK-SAME: dep_modes = [#codir.access_mode<read>, #codir.access_mode<write>]
// CHECK-SAME: logical_worker_slice = [18, 64]
// CHECK-SAME: tile_shape = [18, 64]

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @owner_slice_write_only(%A: memref<128x64xf32>, %C: memref<128x64xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    sde.cu_region <parallel> {
      sde.su_iterate (%c0) to (%c128) step (%c1) classification(<elementwise>) {
      ^bb0(%i: index):
        scf.for %j = %c0 to %c64 step %c1 {
          %v = memref.load %A[%i, %j] : memref<128x64xf32>
          memref.store %v, %C[%i, %j] : memref<128x64xf32>
        }
        sde.yield
      } {iterationTopology = #sde.iteration_topology<owner_strip>, logicalWorkerSlice = [16, 64], pattern = #sde.pattern<uniform>, physicalBlockShape = [16, 64], physicalOwnerDims = [0]}
      sde.yield
    }
    return
  }
}
