// RUN: %carts-compile %s --arts-config %arts_config --start-from create-dbs --pipeline create-dbs --mlir-print-ir-after-all 2>&1 | %FileCheck %s --implicit-check-not="EDT region captures scalar value"

module attributes {arts.runtime_total_nodes = 1 : i64, arts.runtime_total_workers = 8 : i64} {
  func.func private @sink(index)

  func.func @create_dbs_keeps_edt_scalars_local() {
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %span = arith.muli %c1, %c64 : index
    %route = arith.constant -1 : i32

    arts.edt <task> <intranode> route(%route) params(%span : index) {
    ^bb0(%span_arg: index):
      %local_c1 = arith.constant 1 : index
      %local_c64 = arith.constant 64 : index
      %local_span = arith.muli %local_c1, %local_c64 : index
      %idx = arith.addi %span_arg, %local_span : index
      func.call @sink(%idx) : (index) -> ()
      arts.yield
    }
    return
  }
}

// CHECK-LABEL: // -----// IR Dump After CreateDbs
// CHECK-LABEL: func.func @create_dbs_keeps_edt_scalars_local
// CHECK: %[[OUTER:.*]] = arith.muli
// CHECK: arts.edt
// CHECK-SAME: params(
// CHECK-SAME: %[[OUTER]]
// CHECK: ^bb0(
// CHECK-SAME: %[[SPAN_ARG:.*]]: index
// CHECK: %[[LOCAL:.*]] = arith.muli
// CHECK: arith.addi %[[SPAN_ARG]], %[[LOCAL]]
