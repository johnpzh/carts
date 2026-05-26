// RUN: %carts-compile %s --arts-config %arts_config --start-from pre-lowering --pipeline pre-lowering | %FileCheck %s

// Small coarse DBs that become read-only after initialization are legal
// read-mostly dependencies for distributed workers. ARTS lowering should emit
// the runtime duplicate preference hint so remote ranks can cache them instead
// of treating them like ordinary whole-array shared state.

// CHECK-LABEL: func.func @small_readonly_coarse_dep_prefers_duplicate
// CHECK: arts_rt.rec_dep
// CHECK-SAME: dep_flags = array<i32: 1>

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @small_readonly_coarse_dep_prefers_duplicate() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %route = arith.constant -1 : i32
    %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>, <coarse>] route(%route : i32) sizes[%c1] elementType(f64) elementSizes[%c128] {arts.read_only_after_init} : (memref<?xi64>, memref<?xmemref<?xf64>>)
    %acq_guid, %acq_ptr = arts.db_acquire[<in>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<?xf64>>) partitioning(<coarse>), indices[], offsets[%c0], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?xf64>>)
    arts.edt <task> <internode> route(%route) (%acq_ptr) : memref<?xmemref<?xf64>> {
    ^bb0(%dep: memref<?xmemref<?xf64>>):
      %slot = arts.db_ref %dep[%c0] : memref<?xmemref<?xf64>> -> memref<?xf64>
      %v = memref.load %slot[%c0] : memref<?xf64>
      func.call @sink(%v) : (f64) -> ()
      arts.yield
    }
    return
  }

  func.func private @sink(f64)
}
