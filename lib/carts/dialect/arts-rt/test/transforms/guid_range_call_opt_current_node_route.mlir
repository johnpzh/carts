// RUN: %carts-compile %s --pass-pipeline='builtin.module(arts-guid-range-call-opt)' \
// RUN:   | %FileCheck %s

module attributes {arts.runtime_total_nodes = 1 : i64, dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @current_node_route_static_loop(%type: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %route = arith.constant -1 : i32
    scf.for %i = %c0 to %c4 step %c1 {
      %guid = func.call @arts_guid_reserve(%type, %route) : (i32, i32) -> i64
      func.call @use(%guid) : (i64) -> ()
    }
    return
  }

  func.func private @use(i64)
  func.func private @arts_guid_reserve(i32, i32) -> i64
}

// CHECK-LABEL: func.func @current_node_route_static_loop
// CHECK: %[[ROUTE:.*]] = arith.constant -1 : i32
// CHECK: %[[RANGE:.*]] = {{.*}}call @arts_guid_reserve_range
// CHECK-SAME: %[[ROUTE]]
// CHECK: scf.for
// CHECK: %[[FROM_INDEX:.*]] = func.call @arts_guid_from_index
// CHECK: func.call @use(%[[FROM_INDEX]])
// CHECK-NOT: = {{.*}}call @arts_guid_reserve(
