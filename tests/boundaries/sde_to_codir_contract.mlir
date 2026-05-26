// RUN: %carts-compile %s --pass-pipeline='builtin.module(convert-sde-to-codir,verify-codir)' \
// RUN:   | %FileCheck %s --implicit-check-not=sde.

// Per-dialect handoff contract: after `convert-sde-to-codir` runs and
// `verify-codir` accepts the result, no `sde.*` op may survive. The implicit
// check guards the boundary against pattern-level skips that would otherwise
// let SDE ops leak past the dialect frontier silently.

// CHECK-LABEL: func.func @sde_to_codir_contract
// CHECK: codir.codelet
// CHECK: codir.yield

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @sde_to_codir_contract() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %A = memref.alloc() : memref<8xf32>
    sde.cu_region <parallel> {
      sde.su_iterate (%c0) to (%c8) step (%c1) classification(<elementwise>) {
      ^bb0(%i: index):
        %v = memref.load %A[%i] : memref<8xf32>
        memref.store %v, %A[%i] : memref<8xf32>
        sde.yield
      }
      sde.yield
    }
    return
  }
}
