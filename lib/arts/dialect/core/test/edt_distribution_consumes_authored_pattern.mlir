// RUN: %carts-compile %s --arts-config %arts_config --start-from concurrency --pipeline edt-distribution | %FileCheck %s

// Core may complete distribution_kind from runtime topology, but a depPattern
// alone is not an authored distribution_pattern. SDE/Core translation must stamp
// that pattern before Core consumes it.
// CHECK-LABEL: func.func @dep_pattern_not_distribution_pattern
// CHECK: arts.epoch
// CHECK-SAME: depPattern = #arts.dep_pattern<matmul>
// CHECK-SAME: distribution_kind = #arts.distribution_kind<block>
// CHECK-SAME: distribution_version = 1 : i32
// CHECK-NOT: distribution_pattern

// CHECK-LABEL: func.func @parent_authored_distribution_pattern
// CHECK: arts.edt <task> <intranode> route(%{{.*}}) attributes {
// CHECK-SAME: distribution_kind = #arts.distribution_kind<block>
// CHECK-SAME: distribution_pattern = #arts.distribution_pattern<uniform>
// CHECK-SAME: distribution_version = 1 : i32

module attributes {arts.runtime_total_nodes = 1 : i64, arts.runtime_total_workers = 4 : i64, dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @dep_pattern_not_distribution_pattern() -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %route = arith.constant -1 : i32
    arts.edt <parallel> <intranode> route(%route) {
      arts.for(%c0) to(%c8) step(%c1) {
      ^bb0(%iv: index):
      } {depPattern = #arts.dep_pattern<matmul>}
    }
    return %route : i32
  }

  func.func @parent_authored_distribution_pattern() -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %route = arith.constant -1 : i32
    arts.edt <parallel> <intranode> route(%route) attributes {distribution_pattern = #arts.distribution_pattern<uniform>} {
      arts.for(%c0) to(%c8) step(%c1) {
      ^bb0(%iv: index):
      }
    }
    return %route : i32
  }
}
