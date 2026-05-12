// RUN: %carts-compile %s --arts-config %arts_config --start-from post-db-refinement --pipeline pre-lowering | %FileCheck %s

// A dependency-free task may capture an allocation-rooted DB view as state.
// The view must not become a runtime DB dependency: lowering packs the root
// handle through paramv and rematerializes the view in the outlined EDT.

// CHECK-LABEL: func.func private @__arts_edt
// CHECK: arts_rt.edt_param_unpack
// CHECK: llvm.inttoptr
// CHECK: polygeist.pointer2memref
// CHECK: arts_rt.db_gep
// CHECK-LABEL: func.func @captures_external_db_ref_view
// CHECK: arts_rt.edt_create
// CHECK-SAME: depCount(%c0_i32)

module attributes {arts.runtime_total_nodes = 1 : i64, arts.runtime_total_workers = 8 : i64, dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @captures_external_db_ref_view() -> i32 {
    %c-1_i32 = arith.constant -1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 1.000000e+00 : f64

    %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>, <coarse>, <uniform>] route(%c-1_i32 : i32) sizes[%c1] elementType(f64) elementSizes[%c1] : (memref<?xi64>, memref<?xmemref<?xf64>>)
    %view = arts.db_ref %ptr[%c0] : memref<?xmemref<?xf64>> -> memref<?xf64>

    %epoch = arts.epoch {
      arts.edt <task> <intranode> route(%c-1_i32) {
        memref.store %cst, %view[%c0] : memref<?xf64>
      }
    } : i64

    arts.db_free(%guid) : memref<?xi64>
    arts.db_free(%ptr) : memref<?xmemref<?xf64>>
    return %c0_i32 : i32
  }
}
