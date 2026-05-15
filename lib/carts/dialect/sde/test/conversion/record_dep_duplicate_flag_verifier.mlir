// RUN: not %carts-compile %s --pipeline raise-memref-dimensionality 2>&1 | %FileCheck %s

// CHECK: dep_flags entries (2) must match datablocks (1)

module {
  func.func @bad(%db : memref<?xi64>) {
    %edt = arith.constant 1 : i64
    arts_rt.rec_dep %edt(%db : memref<?xi64>) {dep_flags = array<i32: 1, 1>}
    return
  }
}
