// RUN: %carts-compile %s --O3 --arts-config %arts_config \
// RUN:   --start-from openmp-to-arts --pipeline openmp-to-arts \
// RUN:   --mlir-print-ir-after-all 2>&1 \
// RUN:   | %FileCheck %s

// Coarse shared memrefs used by SDE scheduling units should become MU storage
// before the SDE/Core boundary. ConvertSdeToArts then lowers that MU storage
// directly to DB objects; CreateDbs should not be needed for this path.

// CHECK-LABEL: // -----// IR Dump After MemoryUnitMaterialization
// CHECK-LABEL: func.func @coarse_mu_materialization
// CHECK: %[[MU:.*]] = sde.mu_alloc : memref<8xf32>
// CHECK: memref.load %[[MU]][
// CHECK: memref.store {{.*}}, %[[MU]][

// CHECK-LABEL: // -----// IR Dump After ConvertSdeToArts
// CHECK-LABEL: func.func @coarse_mu_materialization
// CHECK: arts.db_alloc
// CHECK-SAME: <coarse>
// CHECK-SAME: elementSizes[%c8]
// CHECK: arts.db_acquire[<inout>]
// CHECK-NOT: sde.mu_alloc

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @coarse_mu_materialization() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %A = memref.alloc() : memref<8xf32>
    sde.cu_region <parallel> scope(<local>) {
      sde.su_iterate (%c0) to (%c8) step (%c1) classification(<elementwise>) {
      ^bb0(%i: index):
        %v = memref.load %A[%i] : memref<8xf32>
        memref.store %v, %A[%i] : memref<8xf32>
        sde.yield
      }
      sde.yield
    }
    return
  }
}
