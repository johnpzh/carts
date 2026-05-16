// RUN: %carts-compile %s --pass-pipeline='builtin.module(verify-codir,convert-codir-to-arts,verify-arts-objects-only)' \
// RUN:   | %FileCheck %s --implicit-check-not="arts.edt <task> <internode>"

// A codelet may carry generic distribution intent before SDE/CODIR can
// materialize physical owner/block storage. CODIR-to-ARTS must not turn that
// into internode placement over coarse raw DBs.

module attributes {arts.runtime_total_nodes = 4 : i64, arts.runtime_total_workers = 64 : i64} {
  func.func @unmaterialized_distributed_intent_stays_local(%A: memref<128xf64>,
                                                           %B: memref<128xf64>,
                                                           %i: index) {
    codir.codelet deps(%A, %B : memref<128xf64>, memref<128xf64>)
      params(%i : index)
      attributes {dep_modes = [#codir.access_mode<read>,
                               #codir.access_mode<write>],
                  distribution_kind = #codir.distribution_kind<blocked>,
                  pattern = #codir.pattern<uniform>} {
    ^bb0(%a: memref<128xf64>, %b: memref<128xf64>, %i_arg: index):
      %v = memref.load %a[%i_arg] : memref<128xf64>
      memref.store %v, %b[%i_arg] : memref<128xf64>
      codir.yield
    }
    func.return
  }
}

// CHECK-LABEL: func.func @unmaterialized_distributed_intent_stays_local
// CHECK: arts.db_alloc
// CHECK-SAME: <coarse>
// CHECK: arts.edt <task> <intranode>
// CHECK-SAME: distribution_kind = #arts.distribution_kind<block>
