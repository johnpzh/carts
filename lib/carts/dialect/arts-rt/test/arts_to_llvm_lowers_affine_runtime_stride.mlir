// RUN: %carts-compile %s --arts-config %arts_config --start-from arts-to-llvm --pipeline arts-to-llvm \
// RUN:   | %FileCheck %s

// Runtime DB stride arithmetic may be created as affine.apply during
// pre-lowering. The arts-to-llvm stage must lower it before LLVM conversion
// rewrites adjacent index values to i64, or LLVM translation sees residual
// unrealized casts.

// CHECK-LABEL: func.func @runtime_stride_affine_lowered
// CHECK-NOT: affine.apply
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: arith.muli

#map = affine_map<()[s0, s1] -> (s0 * s1)>

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @runtime_stride_affine_lowered(%outer: index, %inner: index) -> index {
    %stride = affine.apply #map()[%outer, %inner]
    return %stride : index
  }
}
