// RUN: %carts-compile %s --O3 --arts-config %arts_config \
// RUN:   --start-from openmp-to-arts --pipeline openmp-to-arts \
// RUN:   --mlir-print-ir-after-all 2>&1 \
// RUN:   | awk '/IR Dump After ConvertSdeToArts/,/IR Dump After VerifySdeLowered/' \
// RUN:   | %FileCheck %s

// SDE owns the task/data-shape contract. A tiled owner strip over an external
// writable root must lower to an explicit Core dependency slice before
// CreateDbs has any chance to materialize blocked DBs.

// CHECK-LABEL: func.func @owner_slice_db_control
// CHECK: arts.db_control[<inout>] (%{{[^ ]+}} : memref<128x64xf32>) indices[] offsets[%{{[^]]+}}] sizes[%{{[^]]+}}] : memref<?xi64>
// CHECK: arts.edt <task> <intranode>
// CHECK-SAME: (%{{[^)]*}}) : memref<?xi64>
// CHECK-NOT: sde.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @owner_slice_db_control(%A: memref<128x64xf32>, %C: memref<128x64xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index

    sde.cu_region <parallel> scope(<local>) {
      sde.su_iterate (%c0) to (%c128) step (%c1) classification(<elementwise>) {
      ^bb0(%i: index):
        scf.for %j = %c0 to %c64 step %c1 {
          %v = memref.load %A[%i, %j] : memref<128x64xf32>
          %old = memref.load %C[%i, %j] : memref<128x64xf32>
          %next = arith.addf %old, %v : f32
          memref.store %next, %C[%i, %j] : memref<128x64xf32>
        }
        sde.yield
      } {iterationTopology = #sde.iteration_topology<owner_strip>, logicalWorkerSlice = [16, 64], pattern = #sde.pattern<uniform>, physicalBlockShape = [16, 64], physicalOwnerDims = [0]}
      sde.yield
    }

    return
  }
}
