// RUN: %carts-compile %s --pass-pipeline='builtin.module(convert-sde-to-codir,verify-codir)' | %FileCheck %s --implicit-check-not=sde.cu_codelet

module {
  func.func @sde_sliced_token_to_codir(%mem: memref<8xi32>, %scale: i32) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %token = sde.mu_token <readwrite> %mem[%c0] size[%c4]
      : memref<8xi32> -> !sde.token<memref<4xi32>>

    sde.cu_codelet (%token : !sde.token<memref<4xi32>>)
      captures(%scale : i32) {
    ^bb0(%view: memref<4xi32>, %scale_arg: i32):
      %local_c0 = arith.constant 0 : index
      %value = memref.load %view[%local_c0] : memref<4xi32>
      %updated = arith.addi %value, %scale_arg : i32
      memref.store %updated, %view[%local_c0] : memref<4xi32>
      sde.yield
    }
    return
  }
}

// CHECK-LABEL: func.func @sde_sliced_token_to_codir
// CHECK: sde.mu_token
// CHECK: memref.subview
// CHECK-SAME: memref<8xi32> to memref<4xi32>
// CHECK: codir.codelet deps(%{{.*}} : memref<4xi32>) params(%{{.*}} : i32) attributes {dep_modes = [#codir.access_mode<readwrite>]}
// CHECK: ^bb0(%{{.*}}: memref<4xi32>, %{{.*}}: i32):
// CHECK: memref.load
// CHECK: memref.store
// CHECK: codir.yield
