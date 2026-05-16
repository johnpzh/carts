// RUN: not %carts-compile %s --pass-pipeline='builtin.module(verify-codir)' 2>&1 | %FileCheck %s

module {
  func.func @codir_rejects_arts(%dep: memref<4xf32>) {
    codir.codelet deps(%dep : memref<4xf32>) attributes {dep_modes = [#codir.access_mode<read>]} {
    ^bb0(%arg0: memref<4xf32>):
      %nodes = arts.runtime_query <total_nodes> -> i32
      codir.yield
    }
    return
  }

  func.func @codir_rejects_arts_rt() {
    codir.codelet {
      %c0 = arith.constant 0 : i64
      %pack = arts_rt.edt_param_pack(%c0) : i64 : memref<?xi64>
      codir.yield
    }
    return
  }

  func.func @codir_rejects_llvm() {
    codir.codelet {
      %undef = llvm.mlir.undef : i64
      codir.yield
    }
    return
  }
}

// CHECK: materialized arts operation is not allowed inside codir.codelet
// CHECK: materialized arts_rt operation is not allowed inside codir.codelet
// CHECK: materialized llvm operation is not allowed inside codir.codelet
