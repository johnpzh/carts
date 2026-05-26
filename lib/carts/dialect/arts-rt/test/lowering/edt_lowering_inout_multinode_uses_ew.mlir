// RUN: %carts-compile %s --pass-pipeline='builtin.module(edt-lowering)' | %FileCheck %s

// CHECK-LABEL: func.func @inout_multinode_uses_ordered_ew
// CHECK: arts_rt.rec_dep
// CHECK-SAME: acquire_modes = array<i32: 2>
// CHECK-NOT: acquire_modes = array<i32: 3>

module attributes {
  arts.runtime_total_nodes = 4 : i64,
  arts.runtime_total_workers = 16 : i64,
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<[8, 32]> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i16, dense<[16, 32]> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>,
  llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128",
  llvm.target_triple = "aarch64-unknown-linux-gnu"
} {
  func.func @inout_multinode_uses_ordered_ew() {
    %route = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>, <coarse>] route(%route : i32) sizes[%c1] elementType(i32) elementSizes[%c1] : (memref<?xi64>, memref<?xmemref<?xi32>>)
    %acq_guid, %acq_ptr = arts.db_acquire[<inout>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<?xi32>>) partitioning(<coarse>), indices[], offsets[%c0], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?xi32>>)

    arts.edt <task> <internode> route(%route) (%acq_ptr) : memref<?xmemref<?xi32>> {
    ^bb0(%dep: memref<?xmemref<?xi32>>):
      %payload = arts.db_ref %dep[%c0] : memref<?xmemref<?xi32>> -> memref<?xi32>
      %old = memref.load %payload[%c0] : memref<?xi32>
      %new = arith.addi %old, %old : i32
      memref.store %new, %payload[%c0] : memref<?xi32>
      arts.yield
    }

    arts.db_release(%acq_ptr) : memref<?xmemref<?xi32>>
    return
  }
}
