// RUN: %carts-compile %s --pass-pipeline='builtin.module(convert-sde-to-codir,verify-codir,convert-codir-to-arts)' | %FileCheck %s

module {
  func.func @codir_boundary_placeholders(%dep: memref<4xf32>, %param: index) {
    codir.codelet deps(%dep : memref<4xf32>) params(%param : index) attributes {dep_modes = [#codir.access_mode<read>]} {
    ^bb0(%dep_arg: memref<4xf32>, %param_arg: index):
      codir.yield
    }
    return
  }
}

// CHECK-LABEL: func.func @codir_boundary_placeholders
// CHECK: codir.codelet deps(%{{.*}} : memref<4xf32>) params(%{{.*}} : index) attributes {dep_modes = [#codir.access_mode<read>]}
// CHECK: ^bb0(%{{.*}}: memref<4xf32>, %{{.*}}: index):
// CHECK: codir.yield
