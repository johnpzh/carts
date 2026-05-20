// RUN: %carts-compile %s --arts-config %arts_config --start-from arts-rt-to-llvm --pipeline arts-rt-to-llvm | %FileCheck %s

// Multinode EDT placement is carried by the route operand. RT-to-LLVM lowering
// must preserve a dynamic route in the ARTS hint instead of replacing it with
// the current-node sentinel.

// CHECK-LABEL: func.func @dynamic_route_preserved
// CHECK-SAME: %[[ROUTE:arg[0-9]+]]: i32
// CHECK: llvm.insertvalue %[[ROUTE]], {{.*}}[0] : !llvm.struct<(i32, i64)>
// CHECK-NOT: llvm.insertvalue %c-1_i32
// CHECK: {{func[.]call|call}} @arts_edt_create_with_epoch

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func private @__arts_edt(%argc: i32, %argv: !llvm.ptr, %depc: i32, %depv: !llvm.ptr) {
    return
  }

  func.func @dynamic_route_preserved(%route: i32, %epoch: i64) -> i64 {
    %c0_i32 = arith.constant 0 : i32
    %params = memref.alloca() : memref<0xi64>
    %guid = arts_rt.edt_create(%params : memref<0xi64>) depCount(%c0_i32) route(%route) epoch(%epoch : i64) {arts.create_id = 42 : i64, outlined_func = "__arts_edt"}
    return %guid : i64
  }
}
