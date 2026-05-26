// RUN: not %carts-compile %s --pass-pipeline='builtin.module(edt-lowering)' 2>&1 | %FileCheck %s

// CHECK: partial-reduction split plan was not materialized before ARTS-RT EDT lowering
// CHECK: split owner task count: 1920
// CHECK: split target worker count: 4096
// CHECK: partial-reduction split factor: 3
// CHECK: the ARTS layer must materialize reduction-tile partial DBs and a combine tree before this runtime ABI pass

module attributes {
  arts.runtime_total_nodes = 64 : i64,
  arts.runtime_total_workers = 4096 : i64,
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<[8, 32]> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i16, dense<[16, 32]> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>,
  llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128",
  llvm.target_triple = "aarch64-unknown-linux-gnu"
} {
  func.func @unmaterialized_split_plan_reaches_edt_lowering() {
    %route = arith.constant 0 : i32
    arts.edt <task> <internode> route(%route) attributes {
      partialReduction,
      partialReductionSplitOwnerTaskCount = 1920 : i64,
      partialReductionSplitRequired,
      partialReductionSplitDims = [1],
      partialReductionSplitFactor = 3 : i64,
      partialReductionSplitTargetWorkerCount = 4096 : i64
    } {
    ^bb0:
      arts.yield
    }
    return
  }
}
