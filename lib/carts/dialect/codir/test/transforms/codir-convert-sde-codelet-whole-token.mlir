// RUN: %carts-compile %s --pass-pipeline='builtin.module(convert-sde-to-codir,verify-codir)' | %FileCheck %s --implicit-check-not=sde.cu_codelet

module {
  func.func @sde_whole_token_to_codir(%mem: memref<8xi32>, %scale: i32) {
    %token = sde.mu_token <readwrite> %mem
      : memref<8xi32> -> !sde.token<memref<8xi32>>

    sde.cu_codelet (%token : !sde.token<memref<8xi32>>)
      captures(%scale : i32) {
    ^bb0(%view: memref<8xi32>, %scale_arg: i32):
      %c0 = arith.constant 0 : index
      %value = memref.load %view[%c0] : memref<8xi32>
      %updated = arith.addi %value, %scale_arg : i32
      memref.store %updated, %view[%c0] : memref<8xi32>
      sde.yield
    }
    return
  }
}

// CHECK-LABEL: func.func @sde_whole_token_to_codir
// CHECK: sde.mu_token
// CHECK: codir.codelet deps(%{{.*}} : memref<8xi32>) params(%{{.*}} : i32) attributes {dep_modes = [#codir.access_mode<readwrite>]}
// CHECK: ^bb0(%{{.*}}: memref<8xi32>, %{{.*}}: i32):
// CHECK: memref.load
// CHECK: memref.store
// CHECK: codir.yield
