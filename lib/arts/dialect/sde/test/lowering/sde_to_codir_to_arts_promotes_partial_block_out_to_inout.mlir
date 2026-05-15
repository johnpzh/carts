// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_64t.cfg \
// RUN:   --start-from sde-planning --pipeline create-dbs \
// RUN:   | %FileCheck %s --implicit-check-not=sde. --implicit-check-not=codir.codelet

// Raw blocked stencil layouts are materialized before CreateDbs sees ARTS IR.

// CHECK-LABEL: func.func @partial_block_out_slice
// CHECK: arts.db_alloc
// CHECK: arts.db_alloc
// CHECK: arts.db_acquire[<in>]
// CHECK: arts.db_acquire[<out>]
// CHECK: arts.edt <task> <intranode>
// CHECK: arts.db_ref
// CHECK: memref.load
// CHECK: memref.store

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @partial_block_out_slice() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %A = memref.alloc() : memref<8x8xf64>
    %B = memref.alloc() : memref<8x8xf64>
    sde.cu_region <parallel> scope(<local>) {
      sde.su_iterate (%c1) to (%c7) step (%c1) classification(<stencil>) {
      ^bb0(%i: index):
        scf.for %j = %c1 to %c7 step %c1 {
          %v = memref.load %A[%i, %j] : memref<8x8xf64>
          memref.store %v, %B[%i, %j] : memref<8x8xf64>
        }
        sde.yield
      } {accessMaxOffsets = [1, 1], accessMinOffsets = [-1, -1], iterationTopology = #sde.iteration_topology<owner_strip>, logicalWorkerSlice = [2, 8], ownerDims = [0, 1], pattern = #sde.pattern<stencil_tiling_nd>, physicalBlockShape = [2, 8], physicalHaloShape = [1], physicalOwnerDims = [0], spatialDims = [0, 1], writeFootprint = [1, 1]}
      sde.yield
    }
    memref.dealloc %B : memref<8x8xf64>
    memref.dealloc %A : memref<8x8xf64>
    return
  }
}
