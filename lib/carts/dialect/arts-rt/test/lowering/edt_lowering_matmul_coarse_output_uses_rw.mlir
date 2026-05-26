// RUN: %carts-compile %s --pass-pipeline='builtin.module(edt-lowering)' | %FileCheck %s

// Row-owned matmul EDTs read their output after initializing it inside the
// task, so generic payload-read detection must not force the coarse output DB
// through ordered EW. The matmul pattern plus inPlaceSafe marks the first
// dependency as row-disjoint across EDT instances.

// CHECK-LABEL: func.func @matmul_coarse_output_uses_unordered_rw
// CHECK: arts_rt.rec_dep
// CHECK-SAME: acquire_modes = array<i32: 3, 1, 1>

// CHECK-LABEL: func.func @uniform_inplace_safe_coarse_output_uses_unordered_rw
// CHECK: arts_rt.rec_dep
// CHECK-SAME: acquire_modes = array<i32: 3, 1>

module attributes {
  arts.runtime_total_nodes = 1 : i64,
  arts.runtime_total_workers = 4 : i64,
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>,
  llvm.data_layout = "e-m:e-i64:64-n32:64-S128",
  llvm.target_triple = "x86_64-unknown-linux-gnu"
} {
  func.func @matmul_coarse_output_uses_unordered_rw() {
    %route = arith.constant -1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %zero = arith.constant 0.0 : f64

    %out_guid, %out_ptr = arts.db_alloc[<inout>, <heap>, <write>, <coarse>] route(%route : i32) sizes[%c1] elementType(f64) elementSizes[%c4, %c4] {local_only} : (memref<?xi64>, memref<?xmemref<?x?xf64>>)
    %lhs_guid, %lhs_ptr = arts.db_alloc[<in>, <heap>, <read>, <coarse>] route(%route : i32) sizes[%c1] elementType(f64) elementSizes[%c4, %c4] {local_only} : (memref<?xi64>, memref<?xmemref<?x?xf64>>)
    %rhs_guid, %rhs_ptr = arts.db_alloc[<in>, <heap>, <read>, <coarse>] route(%route : i32) sizes[%c1] elementType(f64) elementSizes[%c4, %c4] {local_only} : (memref<?xi64>, memref<?xmemref<?x?xf64>>)

    %out_acq_guid, %out_acq_ptr = arts.db_acquire[<inout>] (%out_guid : memref<?xi64>, %out_ptr : memref<?xmemref<?x?xf64>>) partitioning(<coarse>), indices[], offsets[%c0], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?x?xf64>>)
    %lhs_acq_guid, %lhs_acq_ptr = arts.db_acquire[<in>] (%lhs_guid : memref<?xi64>, %lhs_ptr : memref<?xmemref<?x?xf64>>) partitioning(<coarse>), indices[], offsets[%c0], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?x?xf64>>)
    %rhs_acq_guid, %rhs_acq_ptr = arts.db_acquire[<in>] (%rhs_guid : memref<?xi64>, %rhs_ptr : memref<?xmemref<?x?xf64>>) partitioning(<coarse>), indices[], offsets[%c0], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?x?xf64>>)

    arts.edt <task> <intranode> route(%route) (%out_acq_ptr, %lhs_acq_ptr, %rhs_acq_ptr) : memref<?xmemref<?x?xf64>>, memref<?xmemref<?x?xf64>>, memref<?xmemref<?x?xf64>> params(%c0 : index) attributes {depPattern = #arts.dep_pattern<matmul>, inPlaceSafe} {
    ^bb0(%out_dep: memref<?xmemref<?x?xf64>>, %lhs_dep: memref<?xmemref<?x?xf64>>, %rhs_dep: memref<?xmemref<?x?xf64>>, %row: index):
      %out_payload = arts.db_ref %out_dep[%c0] : memref<?xmemref<?x?xf64>> -> memref<?x?xf64>
      %lhs_payload = arts.db_ref %lhs_dep[%c0] : memref<?xmemref<?x?xf64>> -> memref<?x?xf64>
      %rhs_payload = arts.db_ref %rhs_dep[%c0] : memref<?xmemref<?x?xf64>> -> memref<?x?xf64>
      memref.store %zero, %out_payload[%row, %c0] : memref<?x?xf64>
      %old = memref.load %out_payload[%row, %c0] : memref<?x?xf64>
      %a = memref.load %lhs_payload[%row, %c0] : memref<?x?xf64>
      %b = memref.load %rhs_payload[%c0, %c0] : memref<?x?xf64>
      %prod = arith.mulf %a, %b : f64
      %sum = arith.addf %old, %prod : f64
      memref.store %sum, %out_payload[%row, %c0] : memref<?x?xf64>
      arts.yield
    }

    arts.db_release(%out_acq_ptr) : memref<?xmemref<?x?xf64>>
    arts.db_release(%lhs_acq_ptr) : memref<?xmemref<?x?xf64>>
    arts.db_release(%rhs_acq_ptr) : memref<?xmemref<?x?xf64>>
    return
  }

  func.func @uniform_inplace_safe_coarse_output_uses_unordered_rw() {
    %route = arith.constant -1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index

    %out_guid, %out_ptr = arts.db_alloc[<inout>, <heap>, <write>, <coarse>] route(%route : i32) sizes[%c1] elementType(f64) elementSizes[%c4] {local_only} : (memref<?xi64>, memref<?xmemref<?xf64>>)
    %in_guid, %in_ptr = arts.db_alloc[<in>, <heap>, <read>, <coarse>] route(%route : i32) sizes[%c1] elementType(f64) elementSizes[%c4] {local_only} : (memref<?xi64>, memref<?xmemref<?xf64>>)

    %out_acq_guid, %out_acq_ptr = arts.db_acquire[<inout>] (%out_guid : memref<?xi64>, %out_ptr : memref<?xmemref<?xf64>>) partitioning(<coarse>), indices[], offsets[%c0], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?xf64>>)
    %in_acq_guid, %in_acq_ptr = arts.db_acquire[<in>] (%in_guid : memref<?xi64>, %in_ptr : memref<?xmemref<?xf64>>) partitioning(<coarse>), indices[], offsets[%c0], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?xf64>>)

    arts.edt <task> <intranode> route(%route) (%out_acq_ptr, %in_acq_ptr) : memref<?xmemref<?xf64>>, memref<?xmemref<?xf64>> params(%c0 : index) attributes {depPattern = #arts.dep_pattern<uniform>, inPlaceSafe} {
    ^bb0(%out_dep: memref<?xmemref<?xf64>>, %in_dep: memref<?xmemref<?xf64>>, %idx: index):
      %out_payload = arts.db_ref %out_dep[%c0] : memref<?xmemref<?xf64>> -> memref<?xf64>
      %in_payload = arts.db_ref %in_dep[%c0] : memref<?xmemref<?xf64>> -> memref<?xf64>
      %old = memref.load %out_payload[%idx] : memref<?xf64>
      %new = memref.load %in_payload[%idx] : memref<?xf64>
      %sum = arith.addf %old, %new : f64
      memref.store %sum, %out_payload[%idx] : memref<?xf64>
      arts.yield
    }

    arts.db_release(%out_acq_ptr) : memref<?xmemref<?xf64>>
    arts.db_release(%in_acq_ptr) : memref<?xmemref<?xf64>>
    return
  }
}
