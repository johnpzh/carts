// RUN: %carts-compile %s --arts-config %arts_config --start-from arts-rt-to-llvm --pipeline arts-rt-to-llvm | %FileCheck %s

// CHECK-LABEL: func.func @undef_lowered_in_arts_rt
// CHECK-NOT: arts.undef
// CHECK: llvm.mlir.undef : i32
// CHECK-LABEL: func.func @pack_undef_float_param
// CHECK-NOT: arith.bitcast
// CHECK: llvm.mlir.undef : i64
// CHECK: memref.store
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<[8, 32]> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i16, dense<[16, 32]> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @undef_lowered_in_arts_rt() -> i32 {
    %v = arts.undef : i32
    return %v : i32
  }

  func.func @pack_undef_float_param() -> memref<?xi64> {
    %v = llvm.mlir.undef : f64
    %pack = arts_rt.edt_param_pack(%v) : f64 : memref<?xi64>
    return %pack : memref<?xi64>
  }
}
