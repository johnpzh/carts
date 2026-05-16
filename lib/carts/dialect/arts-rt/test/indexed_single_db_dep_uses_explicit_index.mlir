// RUN: %carts-compile %s --arts-config %arts_config --start-from pre-lowering --pipeline pre-lowering | %FileCheck %s --check-prefix=PRE --implicit-check-not=arts.db_num_elements
// RUN: %carts-compile %s --arts-config %arts_config --start-from pre-lowering --pipeline arts-rt-to-llvm | %FileCheck %s

// A single-element acquire with an explicit DB index still records a direct
// dependency on that indexed source DB, not on slot zero.

// PRE-LABEL: func.func @indexed_single_db_dependency
// PRE: arts_rt.edt_create

// CHECK-LABEL: func.func @indexed_single_db_dependency
// CHECK-SAME: (%[[LANE:arg[0-9]+]]: index,
// CHECK: %[[LANE64:.*]] = arith.index_cast %[[LANE]] : index to i64
// CHECK: llvm.getelementptr {{.*}}[%[[LANE64]]]
// CHECK: llvm.load
// CHECK: {{func|llvm}}.call @arts_add_dependence

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @indexed_single_db_dependency(%lane: index, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %route = arith.constant -1 : i32
    %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>, <fine_grained>] route(%route : i32) sizes[%n] elementType(i64) elementSizes[%c1] : (memref<?xi64>, memref<?xmemref<?xi64>>)
    %acq_guid, %acq_ptr = arts.db_acquire[<inout>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<?xi64>>) partitioning(<fine_grained>, indices[%lane]), indices[%lane], offsets[%c0], sizes[%c1] {preserve_access_mode = #arts.preserve_access_mode, preserve_dep_edge = #arts.preserve_dep_edge} -> (memref<?xi64>, memref<?xmemref<?xi64>>)
    arts.edt <task> <intranode> route(%route) (%acq_ptr) : memref<?xmemref<?xi64>> {
    ^bb0(%dep: memref<?xmemref<?xi64>>):
      %slot = arts.db_ref %dep[%c0] : memref<?xmemref<?xi64>> -> memref<?xi64>
      memref.store %c0_i64, %slot[%c0] : memref<?xi64>
      arts.yield
    }
    return
  }
}
