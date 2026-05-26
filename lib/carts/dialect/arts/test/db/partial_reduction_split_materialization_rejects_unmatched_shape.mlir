// RUN: not %carts-compile %s --pass-pipeline='builtin.module(partial-reduction-split-materialization)' \
// RUN:   2>&1 | %FileCheck %s

module {
  func.func @partial_reduction_split_materialization_rejects_unmatched_shape() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %route = arith.constant -1 : i32
    scf.for %owner = %c0 to %c4 step %c1 {
      arts.edt <task> <intranode> route(%route)
          params(%owner : index)
          attributes {
            partialReduction,
            partialReductionSplitRequired
          } {
      ^bb0(%owner_arg: index):
      }
    }
    return
  }
}

// CHECK: requires static partialReductionSplitFactor > 1
// CHECK: failed to materialize partial-reduction split plan
