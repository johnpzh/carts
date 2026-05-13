// RUN: %carts-compile %s --arts-config %arts_config --start-from concurrency --pipeline edt-distribution | %FileCheck %s

// DB alignment spans are expressed in induction-value space. When an arts.for
// has already been strip-mined by that span, ForLowering must convert the span
// to one loop iteration instead of chunking span * span elements per task.
//
// CHECK-LABEL: func.func @db_alignment_normalizes_nonunit_step
// CHECK: %[[CLAMP:.*]] = arith.minui %c4{{(_[0-9]+)?}}, %{{.*}} : index
// CHECK: %[[DISPATCH:.*]] = arith.maxui %[[CLAMP]], %{{.*}} : index
// CHECK: arts.epoch
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[DISPATCH]] step %{{.*}}
// CHECK: arts.db_acquire
// CHECK: arts.edt <task>

module attributes {arts.runtime_total_nodes = 1 : i64, arts.runtime_total_workers = 4 : i64, dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @db_alignment_normalizes_nonunit_step() -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %route = arith.constant -1 : i32
    %zero = arith.constant 0.000000e+00 : f64

    %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>, <block>, <uniform>] route(%route : i32) sizes[%c4] elementType(f64) elementSizes[%c4] {depPattern = #arts.dep_pattern<uniform>, distribution_kind = #arts.distribution_kind<block>, distribution_pattern = #arts.distribution_pattern<uniform>, distribution_version = 1 : i32, planIterationTopology = #arts.plan_iteration_topology<owner_strip>, planLogicalWorkerSlice = [4], planOwnerDims = [0], planPhysicalBlockShape = [4]} : (memref<?xi64>, memref<?xmemref<?xf64>>)
    %acq_guid, %acq_ptr = arts.db_acquire[<out>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<?xf64>>) partitioning(<block>), indices[], offsets[%c0], sizes[%c4] {depPattern = #arts.dep_pattern<uniform>, distribution_kind = #arts.distribution_kind<block>, distribution_pattern = #arts.distribution_pattern<uniform>, distribution_version = 1 : i32} -> (memref<?xi64>, memref<?xmemref<?xf64>>)

    arts.edt <parallel> <intranode> route(%route) (%acq_ptr) : memref<?xmemref<?xf64>> attributes {depPattern = #arts.dep_pattern<uniform>, distribution_kind = #arts.distribution_kind<block>, distribution_pattern = #arts.distribution_pattern<uniform>, distribution_version = 1 : i32, planIterationTopology = #arts.plan_iteration_topology<owner_strip>, planLogicalWorkerSlice = [4], planOwnerDims = [0], planPhysicalBlockShape = [4], workers = #arts.workers<4>} {
    ^bb0(%arg0: memref<?xmemref<?xf64>>):
      arts.for(%c0) to(%c16) step(%c4) {
      ^bb0(%i: index):
        %block_idx = arith.divui %i, %c4 : index
        %dst = arts.db_ref %arg0[%block_idx] : memref<?xmemref<?xf64>> -> memref<?xf64>
        memref.store %zero, %dst[%c0] : memref<?xf64>
      } {depPattern = #arts.dep_pattern<uniform>, distribution_kind = #arts.distribution_kind<block>, distribution_pattern = #arts.distribution_pattern<uniform>, distribution_version = 1 : i32, planIterationTopology = #arts.plan_iteration_topology<owner_strip>, planLogicalWorkerSlice = [4], planOwnerDims = [0], planPhysicalBlockShape = [4]}
      arts.db_release(%arg0) : memref<?xmemref<?xf64>>
    }

    arts.db_release(%acq_ptr) : memref<?xmemref<?xf64>>
    arts.db_free(%guid) : memref<?xi64>
    arts.db_free(%ptr) : memref<?xmemref<?xf64>>
    %ret = arith.constant 0 : i32
    return %ret : i32
  }
}
