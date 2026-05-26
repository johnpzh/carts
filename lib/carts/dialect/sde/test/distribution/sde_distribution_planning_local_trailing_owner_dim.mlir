// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_64t.cfg --start-from sde-planning --pipeline codir-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=SDE
// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_64t.cfg --start-from sde-planning --pipeline create-dbs --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=DB

// Imperfect local stencil/update loops can carry owner-slice scheduling intent
// when the local owner IV maps to the trailing physical output dimension, and
// all output self-reads stay within that owner slice. CreateDbs keeps the
// host-visible source allocation coarse and materializes a planned block DB for
// the owner-slice task storage.

// SDE-LABEL: // -----// IR Dump After DistributionPlanning (distribution-planning) //----- //
// SDE: func.func @main
// SDE: sde.su_iterate (%c2) to (%c62) step (%c1) schedule(<static>) classification(<stencil>) {
// SDE: } {
// SDE-SAME: iterationTopology = #sde.iteration_topology<owner_strip>
// SDE-SAME: logicalWorkerSlice = [16, 16, 1]
// SDE-SAME: physicalBlockShape = [16, 16, 1]
// SDE-SAME: physicalOwnerDims = [2]
// SDE-LABEL: // -----// IR Dump After IterationSpaceDecomposition

// DB-LABEL: // -----// IR Dump After CreateDbs
// DB: func.func @main
// DB: arts.db_alloc
// DB-SAME: <coarse>
// DB: arts.db_alloc
// DB-SAME: <block>
// DB-SAME: planOwnerDims = [2]
// DB-SAME: planPhysicalBlockShape = [16, 16, 1]
// DB: arts.edt <task>
// DB-SAME: planOwnerDims = [2]
// DB-SAME: stencil_owner_dims = [0, 1, 2]

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c14 = arith.constant 14 : index
    %c16 = arith.constant 16 : index
    %c62 = arith.constant 62 : index
    %zero = arith.constant 0.000000e+00 : f64
    %A = memref.alloc() : memref<16x16x64xf64>
    %B = memref.alloc() : memref<16x16x64xf64>
    sde.cu_region <parallel> {
      sde.su_iterate (%c2) to (%c62) step (%c1) schedule(<static>) {
      ^bb0(%k: index):
        %scratch = memref.alloca() : memref<f64>
        memref.store %zero, %scratch[] : memref<f64>
        scf.for %j = %c2 to %c14 step %c1 {
          scf.for %i = %c2 to %c14 step %c1 {
            %kp = arith.addi %k, %c1 : index
            %a = memref.load %A[%i, %j, %kp] : memref<16x16x64xf64>
            %old = memref.load %B[%i, %j, %k] : memref<16x16x64xf64>
            %next = arith.addf %old, %a : f64
            memref.store %next, %B[%i, %j, %k] : memref<16x16x64xf64>
          }
        }
        sde.yield
      }
      sde.yield
    }
    memref.dealloc %B : memref<16x16x64xf64>
    memref.dealloc %A : memref<16x16x64xf64>
    return
  }
}
