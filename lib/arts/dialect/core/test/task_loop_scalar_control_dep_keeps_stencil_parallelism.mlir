// RUN: %carts-compile %s --arts-config %arts_config --start-from concurrency --pipeline edt-distribution | %FileCheck %s

// Stencil loop bodies often update a scalar control/status DB in addition to
// their row-owned output DB. That scalar write is not a physical owner-slice
// write and must not trigger the owner-mismatch fallback that collapses the
// whole stencil loop to one worker.
//
// CHECK-LABEL: func.func @scalar_control_dep_keeps_stencil_parallelism
// CHECK: %[[CLAMP:.*]] = arith.minui %c4{{(_[0-9]+)?}}, %{{.*}} : index
// CHECK: %[[DISPATCH:.*]] = arith.maxui %[[CLAMP]], %{{.*}} : index
// CHECK: arts.epoch
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[DISPATCH]] step %{{.*}}
// CHECK: arts.edt <task>

module attributes {arts.runtime_total_nodes = 1 : i64, arts.runtime_total_workers = 4 : i64, dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @scalar_control_dep_keeps_stencil_parallelism() -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %route = arith.constant -1 : i32
    %true = arith.constant true
    %zero = arith.constant 0.000000e+00 : f64

    %flag_guid, %flag_ptr = arts.db_alloc[<inout>, <stack>, <write>, <coarse>, <stencil>] route(%route : i32) sizes[%c1] elementType(i1) elementSizes[%c1] : (memref<?xi64>, memref<?xmemref<?xi1>>)
    %src_guid, %src_ptr = arts.db_alloc[<in>, <heap>, <read>, <block>, <stencil>] route(%route : i32) sizes[%c4] elementType(f64) elementSizes[%c1, %c4] {depPattern = #arts.dep_pattern<stencil_tiling_nd>, distribution_pattern = #arts.distribution_pattern<stencil>, planHaloShape = [1], planIterationTopology = #arts.plan_iteration_topology<owner_strip>, planLogicalWorkerSlice = [1, 4], planOwnerDims = [0], planPhysicalBlockShape = [1, 4], stencil_block_shape = [1], stencil_max_offsets = [1], stencil_min_offsets = [-1], stencil_owner_dims = [0], stencil_spatial_dims = [0, 1], stencil_supported_block_halo, stencil_write_footprint = [1]} : (memref<?xi64>, memref<?xmemref<?x?xf64>>)
    %dst_guid, %dst_ptr = arts.db_alloc[<inout>, <heap>, <write>, <block>, <stencil>] route(%route : i32) sizes[%c4] elementType(f64) elementSizes[%c1, %c4] {depPattern = #arts.dep_pattern<stencil_tiling_nd>, distribution_pattern = #arts.distribution_pattern<stencil>, planHaloShape = [1], planIterationTopology = #arts.plan_iteration_topology<owner_strip>, planLogicalWorkerSlice = [1, 4], planOwnerDims = [0], planPhysicalBlockShape = [1, 4], stencil_block_shape = [1], stencil_max_offsets = [1], stencil_min_offsets = [-1], stencil_owner_dims = [0], stencil_spatial_dims = [0, 1], stencil_supported_block_halo, stencil_write_footprint = [1]} : (memref<?xi64>, memref<?xmemref<?x?xf64>>)

    %flag_acq_guid, %flag_acq_ptr = arts.db_acquire[<out>] (%flag_guid : memref<?xi64>, %flag_ptr : memref<?xmemref<?xi1>>) partitioning(<coarse>), indices[], offsets[%c0], sizes[%c1] {depPattern = #arts.dep_pattern<stencil_tiling_nd>, distribution_pattern = #arts.distribution_pattern<stencil>, stencil_block_shape = [1], stencil_max_offsets = [1], stencil_min_offsets = [-1], stencil_owner_dims = [0], stencil_spatial_dims = [0, 1], stencil_supported_block_halo, stencil_write_footprint = [1]} -> (memref<?xi64>, memref<?xmemref<?xi1>>)
    %src_acq_guid, %src_acq_ptr = arts.db_acquire[<in>] (%src_guid : memref<?xi64>, %src_ptr : memref<?xmemref<?x?xf64>>) partitioning(<block>), indices[], offsets[%c0], sizes[%c4] {depPattern = #arts.dep_pattern<stencil_tiling_nd>, distribution_pattern = #arts.distribution_pattern<stencil>, stencil_block_shape = [1], stencil_max_offsets = [1], stencil_min_offsets = [-1], stencil_owner_dims = [0], stencil_spatial_dims = [0, 1], stencil_supported_block_halo, stencil_write_footprint = [1]} -> (memref<?xi64>, memref<?xmemref<?x?xf64>>)
    %dst_acq_guid, %dst_acq_ptr = arts.db_acquire[<out>] (%dst_guid : memref<?xi64>, %dst_ptr : memref<?xmemref<?x?xf64>>) partitioning(<block>), indices[], offsets[%c0], sizes[%c4] {depPattern = #arts.dep_pattern<stencil_tiling_nd>, distribution_pattern = #arts.distribution_pattern<stencil>, stencil_block_shape = [1], stencil_max_offsets = [1], stencil_min_offsets = [-1], stencil_owner_dims = [0], stencil_spatial_dims = [0, 1], stencil_supported_block_halo, stencil_write_footprint = [1]} -> (memref<?xi64>, memref<?xmemref<?x?xf64>>)

    arts.edt <parallel> <intranode> route(%route) (%flag_acq_ptr, %src_acq_ptr, %dst_acq_ptr) : memref<?xmemref<?xi1>>, memref<?xmemref<?x?xf64>>, memref<?xmemref<?x?xf64>> attributes {depPattern = #arts.dep_pattern<stencil_tiling_nd>, distribution_pattern = #arts.distribution_pattern<stencil>, planHaloShape = [1], planIterationTopology = #arts.plan_iteration_topology<owner_strip>, planLogicalWorkerSlice = [1, 4], planOwnerDims = [0], planPhysicalBlockShape = [1, 4], stencil_block_shape = [1], stencil_max_offsets = [1], stencil_min_offsets = [-1], stencil_owner_dims = [0], stencil_spatial_dims = [0, 1], stencil_supported_block_halo, stencil_write_footprint = [1], workers = #arts.workers<4>} {
    ^bb0(%flag_arg: memref<?xmemref<?xi1>>, %src_arg: memref<?xmemref<?x?xf64>>, %dst_arg: memref<?xmemref<?x?xf64>>):
      arts.for(%c0) to(%c4) step(%c1) {
      ^bb0(%i: index):
        %flag_block = arts.db_ref %flag_arg[%c0] : memref<?xmemref<?xi1>> -> memref<?xi1>
        memref.store %true, %flag_block[%c0] : memref<?xi1>
        %src_block = arts.db_ref %src_arg[%i] : memref<?xmemref<?x?xf64>> -> memref<?x?xf64>
        %val = memref.load %src_block[%c0, %c0] : memref<?x?xf64>
        %sum = arith.addf %val, %zero : f64
        %dst_block = arts.db_ref %dst_arg[%i] : memref<?xmemref<?x?xf64>> -> memref<?x?xf64>
        memref.store %sum, %dst_block[%c0, %c0] : memref<?x?xf64>
      } {depPattern = #arts.dep_pattern<stencil_tiling_nd>, distribution_pattern = #arts.distribution_pattern<stencil>, planHaloShape = [1], planIterationTopology = #arts.plan_iteration_topology<owner_strip>, planLogicalWorkerSlice = [1, 4], planOwnerDims = [0], planPhysicalBlockShape = [1, 4], stencil_block_shape = [1], stencil_max_offsets = [1], stencil_min_offsets = [-1], stencil_owner_dims = [0], stencil_spatial_dims = [0, 1], stencil_supported_block_halo, stencil_write_footprint = [1]}
      arts.db_release(%flag_arg) : memref<?xmemref<?xi1>>
      arts.db_release(%src_arg) : memref<?xmemref<?x?xf64>>
      arts.db_release(%dst_arg) : memref<?xmemref<?x?xf64>>
    }

    arts.db_release(%flag_acq_ptr) : memref<?xmemref<?xi1>>
    arts.db_release(%src_acq_ptr) : memref<?xmemref<?x?xf64>>
    arts.db_release(%dst_acq_ptr) : memref<?xmemref<?x?xf64>>
    arts.db_free(%flag_guid) : memref<?xi64>
    arts.db_free(%flag_ptr) : memref<?xmemref<?xi1>>
    arts.db_free(%src_guid) : memref<?xi64>
    arts.db_free(%src_ptr) : memref<?xmemref<?x?xf64>>
    arts.db_free(%dst_guid) : memref<?xi64>
    arts.db_free(%dst_ptr) : memref<?xmemref<?x?xf64>>
    %ret = arith.constant 0 : i32
    return %ret : i32
  }
}
