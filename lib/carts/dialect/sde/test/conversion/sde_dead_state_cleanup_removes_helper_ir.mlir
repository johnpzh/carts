// RUN: %carts-compile %s --pass-pipeline='builtin.module(sde-dead-state-cleanup)' | %FileCheck %s

module {
  func.func private @dead_helper()

  func.func @main() {
    %c1_i32 = arith.constant 1 : i32
    %slot = memref.alloca() : memref<i32>
    memref.store %c1_i32, %slot[] : memref<i32>
    %dead = memref.load %slot[] : memref<i32>
    return
  }
}

// CHECK-LABEL: func.func @main
// CHECK-NOT: func.func private @dead_helper
// CHECK-NOT: memref.alloca
// CHECK-NOT: memref.store
// CHECK-NOT: memref.load
