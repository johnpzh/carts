// RUN: %carts-compile %s --pass-pipeline='builtin.module(db-lowering,edt-lowering,epoch-lowering,verify-pre-lowered)' \
// RUN:   --arts-config %inputs_dir/arts_8t.cfg \
// RUN:   | %FileCheck %s --implicit-check-not=arts.edt --implicit-check-not=arts.epoch

// Per-dialect handoff contract: after the pre-lowering pipeline
// (`db-lowering,edt-lowering,epoch-lowering`) and `verify-pre-lowered`,
// no high-level scheduler op (`arts.edt`, `arts.epoch`) may survive. The
// implicit checks pin the same invariant `verify-pre-lowered` enforces in
// C++, guarding against pattern-level skips that would let scheduler ops
// leak into ARTS-RT silently.
//
// Note: `arts.db_alloc` and `arts.db_acquire` intentionally survive this
// stage. They carry DB metadata (planOwnerDims, planPhysicalBlockShape,
// arts.create_id, etc.) into `arts-rt-to-llvm`, which is the dialect frontier
// where DB ops are finally rewritten to runtime calls. Asserting their
// absence here would contradict the live `db-lowering` contract (see
// lib/carts/dialect/arts-rt/test/lowering/db_lowering_preserves_plan_attrs.mlir).

// CHECK-LABEL: func.func @arts_to_arts_rt_contract
// CHECK: arts_rt.edt_create
// CHECK: arts_rt.rec_dep

module attributes {
  arts.runtime_total_nodes = 1 : i64,
  arts.runtime_total_workers = 4 : i64,
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">>,
  llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128",
  llvm.target_triple = "aarch64-unknown-linux-gnu"
} {
  func.func @arts_to_arts_rt_contract() {
    %route = arith.constant -1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %value = arith.constant 1.0 : f64

    %guid, %ptr = arts.db_alloc[<out>, <heap>, <write>, <coarse>] route(%route : i32) sizes[%c1] elementType(f64) elementSizes[%c4] {local_only} : (memref<?xi64>, memref<?xmemref<?xf64>>)
    %acq_guid, %acq_ptr = arts.db_acquire[<out>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<?xf64>>) partitioning(<coarse>), indices[], offsets[%c0], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?xf64>>)

    arts.edt <task> <intranode> route(%route) (%acq_ptr) : memref<?xmemref<?xf64>> {
    ^bb0(%dep: memref<?xmemref<?xf64>>):
      %payload = arts.db_ref %dep[%c0] : memref<?xmemref<?xf64>> -> memref<?xf64>
      memref.store %value, %payload[%c0] : memref<?xf64>
      arts.db_release(%dep) : memref<?xmemref<?xf64>>
      arts.yield
    }

    arts.db_release(%acq_ptr) : memref<?xmemref<?xf64>>
    return
  }
}
