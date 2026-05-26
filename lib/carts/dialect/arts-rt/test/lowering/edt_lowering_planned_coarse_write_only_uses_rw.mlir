// RUN: %carts-compile %s --pass-pipeline='builtin.module(edt-lowering)' | %FileCheck %s

// Planned coarse, local, write-only owner-strip EDTs may use unordered
// DB_MODE_RW on a single node. True read-modify-write bodies keep ordered EW.

// CHECK-LABEL: func.func @planned_coarse_inout_write_only_uses_rw
// CHECK: arts_rt.rec_dep
// CHECK-SAME: acquire_modes = array<i32: 3>

// CHECK-LABEL: func.func @planned_coarse_inout_read_keeps_ew
// CHECK: arts_rt.rec_dep
// CHECK-SAME: acquire_modes = array<i32: 2>

module attributes {
  arts.runtime_total_nodes = 1 : i64,
  arts.runtime_total_workers = 4 : i64,
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>,
  llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128",
  llvm.target_triple = "x86_64-unknown-linux-gnu"
} {
  func.func @planned_coarse_inout_write_only_uses_rw() {
    %route = arith.constant -1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %value = arith.constant 1.0 : f32

    %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>, <coarse>] route(%route : i32) sizes[%c1] elementType(f32) elementSizes[%c64] {arts.local_only} : (memref<?xi64>, memref<?xmemref<?xf32>>)
    %acq_guid, %acq_ptr = arts.db_acquire[<inout>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<?xf32>>) partitioning(<coarse>), indices[], offsets[%c0], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?xf32>>)

    arts.edt <task> <intranode> route(%route) (%acq_ptr) : memref<?xmemref<?xf32>> params(%c0 : index) attributes {planIterationTopology = #arts.plan_iteration_topology<owner_strip>, planLogicalWorkerSlice = [16]} {
    ^bb0(%dep: memref<?xmemref<?xf32>>, %base: index):
      %payload = arts.db_ref %dep[%c0] : memref<?xmemref<?xf32>> -> memref<?xf32>
      memref.store %value, %payload[%base] : memref<?xf32>
      arts.yield
    }

    arts.db_release(%acq_ptr) : memref<?xmemref<?xf32>>
    return
  }

  func.func @planned_coarse_inout_read_keeps_ew() {
    %route = arith.constant -1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index

    %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>, <coarse>] route(%route : i32) sizes[%c1] elementType(f32) elementSizes[%c64] {arts.local_only} : (memref<?xi64>, memref<?xmemref<?xf32>>)
    %acq_guid, %acq_ptr = arts.db_acquire[<inout>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<?xf32>>) partitioning(<coarse>), indices[], offsets[%c0], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?xf32>>)

    arts.edt <task> <intranode> route(%route) (%acq_ptr) : memref<?xmemref<?xf32>> params(%c0 : index) attributes {planIterationTopology = #arts.plan_iteration_topology<owner_strip>, planLogicalWorkerSlice = [16]} {
    ^bb0(%dep: memref<?xmemref<?xf32>>, %base: index):
      %payload = arts.db_ref %dep[%c0] : memref<?xmemref<?xf32>> -> memref<?xf32>
      %old = memref.load %payload[%base] : memref<?xf32>
      memref.store %old, %payload[%base] : memref<?xf32>
      arts.yield
    }

    arts.db_release(%acq_ptr) : memref<?xmemref<?xf32>>
    return
  }
}
