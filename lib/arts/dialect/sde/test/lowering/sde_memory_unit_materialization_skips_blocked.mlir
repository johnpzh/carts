// RUN: %carts-compile %s --O3 --arts-config %arts_config \
// RUN:   --start-from openmp-to-arts --pipeline openmp-to-arts \
// RUN:   --mlir-print-ir-after-all 2>&1 \
// RUN:   | %FileCheck %s

// Blocked owner-slice layouts materialize through SDE MU storage before the
// boundary. ConvertSdeToArts then consumes the SDE owner-slice facts directly:
// it creates a block-partitioned acquire for dependency slicing while the EDT
// body indexes the full MU payload directly. This keeps the boundary free of
// DB-payload memref.subview rewrites.

// CHECK-LABEL: // -----// IR Dump After MemoryUnitMaterialization
// CHECK-LABEL: func.func @blocked_mu_materialization_owner_slice
// CHECK: %[[MU:.*]] = sde.mu_alloc : memref<8x8xf32>
// CHECK: sde.su_iterate
// CHECK: physicalBlockShape = [4, 8]
// CHECK-LABEL: // -----// IR Dump After ConvertSdeToArts
// CHECK-LABEL: func.func @blocked_mu_materialization_owner_slice
// CHECK: arts.db_alloc
// CHECK-SAME: <coarse>
// CHECK: arts.db_acquire[<inout>]
// CHECK-SAME: partitioning(<block>, offsets[
// CHECK-SAME: sizes[
// CHECK-NOT: memref.subview
// CHECK: memref.load %{{.*}}[{{.*}}, {{.*}}] : memref<8x8xf32>
// CHECK: memref.store {{.*}}, %{{.*}}[{{.*}}, {{.*}}] : memref<8x8xf32>
// CHECK-NOT: sde.mu_alloc

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @blocked_mu_materialization_owner_slice() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %A = memref.alloc() : memref<8x8xf32>
    sde.cu_region <parallel> scope(<local>) {
      sde.su_iterate (%c0) to (%c8) step (%c1) classification(<elementwise>) {
      ^bb0(%i: index):
        scf.for %j = %c0 to %c8 step %c1 {
          %v = memref.load %A[%i, %j] : memref<8x8xf32>
          memref.store %v, %A[%i, %j] : memref<8x8xf32>
        }
        sde.yield
      } {iterationTopology = #sde.iteration_topology<owner_strip>, logicalWorkerSlice = [4, 8], physicalBlockShape = [4, 8], physicalOwnerDims = [0]}
      sde.yield
    }
    return
  }
}
