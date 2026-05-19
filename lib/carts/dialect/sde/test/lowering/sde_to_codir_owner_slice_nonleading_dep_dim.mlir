// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_multinode_4x16.cfg \
// RUN:   --start-from sde-planning --pipeline sde-to-codir \
// RUN:   | %FileCheck %s

// Owner-local pipeline codelets may write a rank-1 owner result while reading
// a higher-rank tensor whose owner index is not the leading physical dimension.
// SDE-to-CODIR should preserve both dependencies as compute-block candidates;
// CODIR-to-ARTS later infers the per-dependency physical owner dimension.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @owner_pipeline_nonleading_dep_dim(%A: memref<8x16x4xf32>, %mean: memref<16xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %zero = arith.constant 0.0 : f32
    sde.cu_region <parallel> {
      sde.su_iterate (%c0) to (%c16) step (%c1) classification(<elementwise_pipeline>) {
      ^bb0(%j: index):
        memref.store %zero, %mean[%j] : memref<16xf32>
        scf.for %b = %c0 to %c8 step %c1 {
          scf.for %k = %c0 to %c4 step %c1 {
            %v = memref.load %A[%b, %j, %k] : memref<8x16x4xf32>
            %old = memref.load %mean[%j] : memref<16xf32>
            %next = arith.addf %old, %v : f32
            memref.store %next, %mean[%j] : memref<16xf32>
          }
        }
        sde.yield
      } {distributionKind = #sde.distribution_kind<blocked>, iterationTopology = #sde.iteration_topology<owner_strip>, logicalWorkerSlice = [4], pattern = #sde.pattern<elementwise_pipeline>, physicalBlockShape = [4], physicalOwnerDims = [0]}
      sde.yield
    }
    return
  }
}

// CHECK-LABEL: func.func @owner_pipeline_nonleading_dep_dim
// CHECK: codir.codelet
// CHECK-SAME: dep_storage_views = [#codir.storage_view<compute_block>, #codir.storage_view<compute_block>]
// CHECK-SAME: logical_worker_slice = [{{[0-9]+}}]
// CHECK-SAME: tile_owner_dims = [0]
// CHECK-SAME: tile_shape = [{{[0-9]+}}]
