// RUN: not %carts-compile %s --arts-config %arts_config --pipeline sde-input-normalization 2>&1 | %FileCheck %s

// CHECK: 'arith.addi' op using value defined outside the region

module {
  func.func @codir_rejects_implicit_capture(%n: index) {
    %c1 = arith.constant 1 : index
    codir.codelet {
      %sum = arith.addi %n, %c1 : index
      codir.yield %sum : index
    }
    return
  }
}
