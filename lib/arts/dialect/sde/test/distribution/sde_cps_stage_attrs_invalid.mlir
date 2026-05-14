// RUN: not %carts-compile %s --arts-config %arts_config --start-from initial-cleanup --pipeline initial-cleanup 2>&1 | %FileCheck %s

// CHECK: sde.cps stage plan requires sde.cps_group_id, sde.cps_stage_index, and sde.cps_stage_count together

module {
  func.func @missing_cps_stage_attrs() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    arts_sde.su_iterate (%c0) to (%c1) step (%c1) {
      arts_sde.yield
    } {cps_group_id = 0 : i64}
    return
  }
}
