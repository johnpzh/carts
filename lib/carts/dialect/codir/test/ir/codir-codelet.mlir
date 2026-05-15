// RUN: %carts-compile %s --arts-config %arts_config --pipeline raise-memref-dimensionality | %FileCheck %s

// CHECK-LABEL: module
// CHECK: codir.codelet
// CHECK-SAME: deps(%{{.*}} : memref<4xf32>)
// CHECK-SAME: params(%{{.*}} : index)
// CHECK: codir.yield

module {
  func.func @codir_parse(%arg0: memref<4xf32>, %n: index) {
    codir.codelet deps(%arg0 : memref<4xf32>) params(%n : index) {
      codir.yield
    }
    return
  }
}
