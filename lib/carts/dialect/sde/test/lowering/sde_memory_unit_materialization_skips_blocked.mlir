// RUN: %carts-compile %s --pass-pipeline='builtin.module(convert-sde-to-codir,verify-codir,convert-codir-to-arts)' \
// RUN:   | %FileCheck %s --implicit-check-not=sde. --implicit-check-not=codir.codelet

// Blocked owner-slice layouts materialize through SDE MU storage, CODIR
// codelets, and then ARTS DB/EDT objects. The 4800x4800 case must keep the SDE
// physical block shape and emit row-strip DB blocks, not one coarse DB.

// CHECK-LABEL: func.func @blocked_mu_materialization_owner_slice
// CHECK: arts.db_alloc
// CHECK-SAME: elementSizes[%c4, %c8{{(_[0-9]+)?}}]
// CHECK-SAME: planPhysicalBlockShape = [4, 8]
// CHECK: arts.db_acquire
// CHECK-SAME: partitioning(<block>)
// CHECK: arts.edt <task> <intranode>
// CHECK: arith.subi %{{.*}}, %{{.*}} : index

// CHECK-LABEL: func.func @blocked_mu_materialization_large64_row_strip
// CHECK: arts.db_alloc
// CHECK-SAME: elementSizes[%c75, %c4800{{(_[0-9]+)?}}]
// CHECK-SAME: planPhysicalBlockShape = [75, 4800]
// CHECK: %[[REL:.*]] = arith.subi %arg0, %{{.*}} : index
// CHECK: arith.divui %[[REL]], %c75{{(_[0-9]+)?}} : index
// CHECK: arts.db_acquire
// CHECK-SAME: partitioning(<block>)
// CHECK: arts.edt <task> <intranode>
// CHECK: arith.subi %{{.*}}, %{{.*}} : index

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @blocked_mu_materialization_owner_slice() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %A = memref.alloc() : memref<8x8xf32>
    sde.cu_region <parallel> {
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

  func.func @blocked_mu_materialization_large64_row_strip() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4800 = arith.constant 4800 : index
    %A = memref.alloc() : memref<4800x4800xf32>
    sde.cu_region <parallel> {
      sde.su_iterate (%c0) to (%c4800) step (%c1) classification(<elementwise>) {
      ^bb0(%i: index):
        scf.for %j = %c0 to %c4800 step %c1 {
          %v = memref.load %A[%i, %j] : memref<4800x4800xf32>
          memref.store %v, %A[%i, %j] : memref<4800x4800xf32>
        }
        sde.yield
      } {iterationTopology = #sde.iteration_topology<owner_strip>, logicalWorkerSlice = [75, 4800], physicalBlockShape = [75, 4800], physicalOwnerDims = [0]}
      sde.yield
    }
    return
  }
}
