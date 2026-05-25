// RUN: %carts-compile %s --pass-pipeline='builtin.module(sde-memory-unit-materialization,convert-sde-to-codir,verify-codir)' \
// RUN:   | %FileCheck %s --implicit-check-not=sde.

// Owner-local, reduction-shaped pipelines with no cross-task reduction
// accumulator may use owner-slice storage. Scratch allocations whose only uses
// are inside the scheduling unit must stay codelet-local rather than becoming
// host-whole readwrite dependencies that serialize all workers.

// CHECK-LABEL: func.func @owner_local_reduction_keeps_block_plan
// CHECK: codir.codelet
// CHECK-SAME: deps(%{{[^,]+}}, %{{[^ ]+}} :
// CHECK-SAME: dep_modes = [#codir.access_mode<read>, #codir.access_mode<write>]
// CHECK-SAME: dep_storage_views = [#codir.storage_view<compute_block>, #codir.storage_view<compute_block>]
// CHECK-SAME: pattern = #codir.pattern<reduction>
// CHECK-SAME: tile_owner_dims = [0]
// CHECK-SAME: tile_shape = [4, 8]
// CHECK: memref.alloca() : memref<8xf32>
// CHECK: codir.yield

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @owner_local_reduction_keeps_block_plan() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %in = memref.alloc() : memref<8x8xf32>
    %out = memref.alloc() : memref<8x8xf32>
    sde.cu_region <parallel> {
      %scratch = memref.alloca() : memref<8xf32>
      sde.su_iterate (%c0) to (%c8) step (%c1) classification(<reduction>) {
      ^bb0(%i: index):
        scf.for %j = %c0 to %c8 step %c1 {
          %v = memref.load %in[%i, %j] : memref<8x8xf32>
          memref.store %v, %scratch[%j] : memref<8xf32>
        }
        scf.for %j = %c0 to %c8 step %c1 {
          %v = memref.load %scratch[%j] : memref<8xf32>
          memref.store %v, %out[%i, %j] : memref<8x8xf32>
        }
        sde.yield
      } {iterationTopology = #sde.iteration_topology<owner_strip>, logicalWorkerSlice = [4, 8], pattern = #sde.pattern<reduction>, physicalBlockShape = [4, 8], physicalOwnerDims = [0]}
      sde.yield
    }
    return
  }
}
