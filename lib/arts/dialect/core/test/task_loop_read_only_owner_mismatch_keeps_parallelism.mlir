// RUN: %carts-compile %s --arts-config %arts_config --start-from concurrency --pipeline post-distribution-cleanup | %FileCheck %s

// A channel-owned reduction may read a tensor whose physical DB layout is
// batch-owned. The read-only tensor must not force channel chunks to align to
// the full channel extent, and each worker must keep the full parent read
// dependency window instead of fabricating a batch-owned slice from the channel
// IV.
//
// CHECK-LABEL: func.func @read_only_owner_mismatch_keeps_parallelism
// CHECK: %[[OUT_GUID:[^,]+]], %[[OUT_PTR:[^ ]+]] = arts.db_alloc
// CHECK: %[[MEAN_GUID:[^,]+]], %[[MEAN_PTR:[^ ]+]] = arts.db_alloc
// CHECK: arts.epoch
// CHECK: scf.for %[[W:arg[0-9]+]] = %{{.*}} to %c4 step %c1
// CHECK: arts.db_acquire[<inout>] (%[[MEAN_GUID]] : memref<?xi64>, %[[MEAN_PTR]] : memref<?xmemref<?xf32>>)
// CHECK-SAME: offsets[%{{.*}}], sizes[%{{.*}}]
// CHECK: arts.db_acquire[<in>] (%[[OUT_GUID]] : memref<?xi64>, %[[OUT_PTR]] : memref<?xmemref<?x?x?xf32>>)
// CHECK-SAME: offsets[%c0], sizes[%c4]
// CHECK: arts.edt <task>

module attributes {arts.runtime_total_nodes = 1 : i64, arts.runtime_total_workers = 4 : i64, dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @read_only_owner_mismatch_keeps_parallelism() -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %route = arith.constant -1 : i32
    %zero = arith.constant 0.000000e+00 : f32

    %out_guid, %out_ptr = arts.db_alloc[<inout>, <heap>, <write>, <block>, <uniform>] route(%route : i32) sizes[%c4] elementType(f32) elementSizes[%c2, %c8, %c4] {depPattern = #arts.dep_pattern<uniform>, distribution_kind = #arts.distribution_kind<block>, distribution_pattern = #arts.distribution_pattern<uniform>, distribution_version = 1 : i32, planIterationTopology = #arts.plan_iteration_topology<owner_strip>, planLogicalWorkerSlice = [2, 8, 4], planOwnerDims = [0], planPhysicalBlockShape = [2, 8, 4]} : (memref<?xi64>, memref<?xmemref<?x?x?xf32>>)
    %mean_guid, %mean_ptr = arts.db_alloc[<inout>, <heap>, <write>, <block>, <uniform>] route(%route : i32) sizes[%c4] elementType(f32) elementSizes[%c2] {depPattern = #arts.dep_pattern<uniform>, distribution_kind = #arts.distribution_kind<block>, distribution_pattern = #arts.distribution_pattern<uniform>, distribution_version = 1 : i32, planIterationTopology = #arts.plan_iteration_topology<owner_strip>, planLogicalWorkerSlice = [2], planOwnerDims = [0], planPhysicalBlockShape = [2]} : (memref<?xi64>, memref<?xmemref<?xf32>>)

    %mean_acq_guid, %mean_acq_ptr = arts.db_acquire[<inout>] (%mean_guid : memref<?xi64>, %mean_ptr : memref<?xmemref<?xf32>>) partitioning(<block>), indices[], offsets[%c0], sizes[%c4] -> (memref<?xi64>, memref<?xmemref<?xf32>>)
    %out_acq_guid, %out_acq_ptr = arts.db_acquire[<in>] (%out_guid : memref<?xi64>, %out_ptr : memref<?xmemref<?x?x?xf32>>) partitioning(<block>), indices[], offsets[%c0], sizes[%c4] -> (memref<?xi64>, memref<?xmemref<?x?x?xf32>>)

    arts.edt <parallel> <intranode> route(%route) (%mean_acq_ptr, %out_acq_ptr) : memref<?xmemref<?xf32>>, memref<?xmemref<?x?x?xf32>> attributes {depPattern = #arts.dep_pattern<uniform>, distribution_kind = #arts.distribution_kind<block>, distribution_pattern = #arts.distribution_pattern<uniform>, distribution_version = 1 : i32, planIterationTopology = #arts.plan_iteration_topology<owner_strip>, planLogicalWorkerSlice = [2], planOwnerDims = [0], planPhysicalBlockShape = [2]} {
    ^bb0(%mean_arg: memref<?xmemref<?xf32>>, %out_arg: memref<?xmemref<?x?x?xf32>>):
      arts.for(%c0) to(%c8) step(%c1) {
      ^bb0(%channel: index):
        %mean_block_idx = arith.divui %channel, %c2 : index
        %mean_local = arith.remui %channel, %c2 : index
        %mean_block = arts.db_ref %mean_arg[%mean_block_idx] : memref<?xmemref<?xf32>> -> memref<?xf32>
        memref.store %zero, %mean_block[%mean_local] : memref<?xf32>
        scf.for %batch = %c0 to %c8 step %c1 {
          %batch_block_idx = arith.divui %batch, %c2 : index
          %batch_local = arith.remui %batch, %c2 : index
          %out_block = arts.db_ref %out_arg[%batch_block_idx] : memref<?xmemref<?x?x?xf32>> -> memref<?x?x?xf32>
          scf.for %spatial = %c0 to %c4 step %c1 {
            %val = memref.load %out_block[%batch_local, %channel, %spatial] : memref<?x?x?xf32>
            %acc = memref.load %mean_block[%mean_local] : memref<?xf32>
            %sum = arith.addf %acc, %val : f32
            memref.store %sum, %mean_block[%mean_local] : memref<?xf32>
          }
        }
      } {depPattern = #arts.dep_pattern<uniform>, distribution_kind = #arts.distribution_kind<block>, distribution_pattern = #arts.distribution_pattern<uniform>, distribution_version = 1 : i32, planIterationTopology = #arts.plan_iteration_topology<owner_strip>, planLogicalWorkerSlice = [2], planOwnerDims = [0], planPhysicalBlockShape = [2]}
      arts.db_release(%mean_arg) : memref<?xmemref<?xf32>>
      arts.db_release(%out_arg) : memref<?xmemref<?x?x?xf32>>
    }

    arts.db_release(%mean_acq_ptr) : memref<?xmemref<?xf32>>
    arts.db_release(%out_acq_ptr) : memref<?xmemref<?x?x?xf32>>
    arts.db_free(%out_guid) : memref<?xi64>
    arts.db_free(%out_ptr) : memref<?xmemref<?x?x?xf32>>
    arts.db_free(%mean_guid) : memref<?xi64>
    arts.db_free(%mean_ptr) : memref<?xmemref<?xf32>>
    %ret = arith.constant 0 : i32
    return %ret : i32
  }
}
