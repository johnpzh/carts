// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_64t.cfg --start-from sde-planning --pipeline sde-to-codir --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// DistributionPlanning can author a physical owner plan after SDE tiling has
// already widened the owner-loop step. The physical DB block for the owner
// dimension must cover that existing step; otherwise one EDT can index across
// multiple DB blocks after acquiring only one owner dependency.

// CHECK-LABEL: // -----// IR Dump After DistributionPlanning (distribution-planning) //----- //
// CHECK: func.func @trailing_owner_step_expression
// CHECK: sde.su_distribute <blocked>
// CHECK: sde.su_iterate
// CHECK-SAME: classification(<elementwise>)
// CHECK: iterationTopology = #sde.iteration_topology<owner_strip>
// CHECK-SAME: logicalWorkerSlice = [4, 64, 8]
// CHECK-SAME: physicalBlockShape = [4, 64, 8]
// CHECK-SAME: physicalOwnerDims = [1]

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @trailing_owner_step_expression(%A: memref<4x512x8xf32>, %B: memref<4x512x8xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c512 = arith.constant 512 : index
    %balanced = arith.ceildivui %c512, %c8 : index
    %preferred = arith.maxui %balanced, %c1 : index
    %tile = arith.minui %preferred, %c512 : index
    %step = arith.muli %c1, %tile : index
    sde.cu_region <parallel> {
      sde.su_iterate (%c0) to (%c512) step (%step) classification(<elementwise>) {
      ^bb0(%channel: index):
        scf.for %b = %c0 to %c4 step %c1 {
          scf.for %hw = %c0 to %c8 step %c1 {
            %v = memref.load %A[%b, %channel, %hw] : memref<4x512x8xf32>
            memref.store %v, %B[%b, %channel, %hw] : memref<4x512x8xf32>
          }
        }
        sde.yield
      }
      sde.yield
    }
    return
  }
}
