// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_multinode_8x64.cfg \
// RUN:   --start-from sde-planning --pipeline sde-to-codir --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Uniform high-node work should not collapse to one owner element per task.
// The SDE distribution planner must keep the cost-model minimum owner grain so
// later ARTS localization of host-visible storage bridges does not flood the
// runtime with tiny tasks.

// CHECK-LABEL: // -----// IR Dump After DistributionPlanning (distribution-planning) //----- //
// CHECK: func.func @batch_copy_grain_floor
// CHECK: sde.su_distribute <blocked>
// CHECK: sde.su_iterate
// CHECK-SAME: classification(<elementwise>)
// CHECK: iterationTopology = #sde.iteration_topology<owner_strip>
// CHECK-SAME: physicalBlockShape = [3, 1920, 196]
// CHECK-SAME: physicalOwnerDims = [0]

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @batch_copy_grain_floor(%A: memref<1920x1920x196xf32>, %B: memref<1920x1920x196xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c196 = arith.constant 196 : index
    %c1920 = arith.constant 1920 : index
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%b) : index = (%c0) to (%c1920) step (%c1) {
          scf.for %c = %c0 to %c1920 step %c1 {
            scf.for %hw = %c0 to %c196 step %c1 {
              %v = memref.load %A[%b, %c, %hw] : memref<1920x1920x196xf32>
              memref.store %v, %B[%b, %c, %hw] : memref<1920x1920x196xf32>
            }
          }
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }
}
