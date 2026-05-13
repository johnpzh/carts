// RUN: %carts-compile %samples_dir/jacobi/for/jacobi-for.mlir --O3 --arts-config %arts_config --pipeline edt-distribution | %FileCheck %s
// RUN: %carts-compile %s --arts-config %arts_config --start-from concurrency --pipeline edt-distribution | %FileCheck %s --check-prefix=MODE

// Verify that EdtTaskLoopLowering unconditionally computes aligned bounds
// (diffLower/diffUpper via arith.subi, ceilDiv via arith.divui, clamping via
// arith.minui) regardless of the useAlignedLowerBound flag.
// CHECK: arith.subi
// CHECK: arith.divui
// CHECK: arith.minui

// A coarse parent `<inout>` acquire is the conservative DB contract, but the
// worker task should use the mode implied by loop-local memory effects.
//
// MODE-LABEL: func.func @refine_inout_by_loop_local_use
// MODE: %[[IN_GUID:[^,]+]], %[[IN_PTR:[^ ]+]] = arts.db_alloc
// MODE: %[[OUT_GUID:[^,]+]], %[[OUT_PTR:[^ ]+]] = arts.db_alloc
// MODE: arts.db_acquire[<inout>] (%[[IN_GUID]] : memref<?xi64>, %[[IN_PTR]] : memref<?xmemref<?xf64>>)
// MODE: arts.db_acquire[<inout>] (%[[OUT_GUID]] : memref<?xi64>, %[[OUT_PTR]] : memref<?xmemref<?xf64>>)
// MODE-DAG: arts.db_acquire[<in>] (%[[IN_GUID]] : memref<?xi64>, %[[IN_PTR]] : memref<?xmemref<?xf64>>)
// MODE-DAG: arts.db_acquire[<out>] (%[[OUT_GUID]] : memref<?xi64>, %[[OUT_PTR]] : memref<?xmemref<?xf64>>)
//
// When SDE/CreateDbs have already materialized a block DB, Core must align
// task chunks to that physical block span. Dividing the span by workers again
// creates boundary-sharing DB dependencies and serializes otherwise independent
// owner-strip writers.
//
// MODE-LABEL: func.func @physical_db_block_span_alignment
// MODE-DAG: %[[C63:.*]] = arith.constant 63 : index
// MODE-NOT: arith.muli %{{.*}}, %c2
// MODE-NOT: arith.muli %{{.*}}, %c4
// MODE: arith.muli %{{.*}}, %[[C63]] : index
// MODE: arts.db_acquire
// MODE-SAME: partitioning(<block>
// MODE-SAME: element_sizes

module attributes {arts.runtime_total_nodes = 1 : i64, arts.runtime_total_workers = 8 : i64, dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @refine_inout_by_loop_local_use() -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant 2.000000e+00 : f64

    %in_guid, %in_ptr = arts.db_alloc[<inout>, <heap>, <write>, <coarse>, <uniform>] route(%c-1_i32 : i32) sizes[%c1] elementType(f64) elementSizes[%c16] : (memref<?xi64>, memref<?xmemref<?xf64>>)
    %out_guid, %out_ptr = arts.db_alloc[<inout>, <heap>, <write>, <coarse>, <uniform>] route(%c-1_i32 : i32) sizes[%c1] elementType(f64) elementSizes[%c16] : (memref<?xi64>, memref<?xmemref<?xf64>>)

    %in_acq_guid, %in_acq_ptr = arts.db_acquire[<inout>] (%in_guid : memref<?xi64>, %in_ptr : memref<?xmemref<?xf64>>) partitioning(<coarse>), indices[], offsets[%c0], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?xf64>>)
    %out_acq_guid, %out_acq_ptr = arts.db_acquire[<inout>] (%out_guid : memref<?xi64>, %out_ptr : memref<?xmemref<?xf64>>) partitioning(<coarse>), indices[], offsets[%c0], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?xf64>>)

    arts.edt <parallel> <intranode> route(%c-1_i32) (%in_acq_ptr, %out_acq_ptr) : memref<?xmemref<?xf64>>, memref<?xmemref<?xf64>> {
    ^bb0(%arg0: memref<?xmemref<?xf64>>, %arg1: memref<?xmemref<?xf64>>):
      arts.for(%c0) to(%c16) step(%c1) {
      ^bb0(%iv: index):
        %src = arts.db_ref %arg0[%c0] : memref<?xmemref<?xf64>> -> memref<?xf64>
        %val = memref.load %src[%iv] : memref<?xf64>
        %scaled = arith.mulf %val, %cst : f64
        %dst = arts.db_ref %arg1[%c0] : memref<?xmemref<?xf64>> -> memref<?xf64>
        memref.store %scaled, %dst[%iv] : memref<?xf64>
      }
      arts.db_release(%arg0) : memref<?xmemref<?xf64>>
      arts.db_release(%arg1) : memref<?xmemref<?xf64>>
    }

    arts.db_release(%in_acq_ptr) : memref<?xmemref<?xf64>>
    arts.db_release(%out_acq_ptr) : memref<?xmemref<?xf64>>
    arts.db_free(%in_guid) : memref<?xi64>
    arts.db_free(%in_ptr) : memref<?xmemref<?xf64>>
    arts.db_free(%out_guid) : memref<?xi64>
    arts.db_free(%out_ptr) : memref<?xmemref<?xf64>>
    %ret = arith.constant 0 : i32
    return %ret : i32
  }

  func.func @physical_db_block_span_alignment() -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c63 = arith.constant 63 : index
    %c1000 = arith.constant 1000 : index
    %c-1_i32 = arith.constant -1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f64

    %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>, <block>, <indexed>] route(%c-1_i32 : i32) sizes[%c16] elementType(f64) elementSizes[%c63] {planIterationTopology = #arts.plan_iteration_topology<owner_strip>, planOwnerDims = [0], planPhysicalBlockShape = [63]} : (memref<?xi64>, memref<?xmemref<?xf64>>)
    %soft_guid, %soft_ptr = arts.db_alloc[<inout>, <heap>, <write>, <coarse>, <uniform>] route(%c-1_i32 : i32) sizes[%c1] elementType(f64) elementSizes[%c16] : (memref<?xi64>, memref<?xmemref<?xf64>>)
    %acq_guid, %acq_ptr = arts.db_acquire[<out>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<?xf64>>) partitioning(<block>), indices[], offsets[%c0], sizes[%c16] -> (memref<?xi64>, memref<?xmemref<?xf64>>)
    %soft_acq_guid, %soft_acq_ptr = arts.db_acquire[<inout>] (%soft_guid : memref<?xi64>, %soft_ptr : memref<?xmemref<?xf64>>) partitioning(<coarse>), indices[], offsets[%c0], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?xf64>>)
    arts.edt <parallel> <intranode> route(%c-1_i32) (%acq_ptr, %soft_acq_ptr) : memref<?xmemref<?xf64>>, memref<?xmemref<?xf64>> {
    ^bb0(%arg0: memref<?xmemref<?xf64>>, %arg1: memref<?xmemref<?xf64>>):
      arts.for(%c0) to(%c1000) step(%c1) {
      ^bb0(%iv: index):
        %block = arith.divui %iv, %c63 : index
        %elem = arith.remui %iv, %c63 : index
        %dst = arts.db_ref %arg0[%block] : memref<?xmemref<?xf64>> -> memref<?xf64>
        memref.store %cst, %dst[%elem] : memref<?xf64>
      } {planIterationTopology = #arts.plan_iteration_topology<owner_strip>, planOwnerDims = [0], planPhysicalBlockShape = [63], distribution_pattern = #arts.distribution_pattern<uniform>}
      arts.db_release(%arg0) : memref<?xmemref<?xf64>>
      arts.db_release(%arg1) : memref<?xmemref<?xf64>>
    }

    arts.db_release(%acq_ptr) : memref<?xmemref<?xf64>>
    arts.db_release(%soft_acq_ptr) : memref<?xmemref<?xf64>>
    arts.db_free(%guid) : memref<?xi64>
    arts.db_free(%ptr) : memref<?xmemref<?xf64>>
    arts.db_free(%soft_guid) : memref<?xi64>
    arts.db_free(%soft_ptr) : memref<?xmemref<?xf64>>
    return %c0_i32 : i32
  }
}
