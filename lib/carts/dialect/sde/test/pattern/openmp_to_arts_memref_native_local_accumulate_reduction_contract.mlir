// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_2t.cfg --start-from sde-planning --pipeline sde-planning --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Verify that the broader nested reduction case stays memref-native:
// classification and strategy are preserved, but no transient carrier is
// materialized.

// CHECK-LABEL: // -----// IR Dump After PatternAnalysis (sde-pattern-analysis) //----- //
// CHECK: func.func @main
// CHECK: sde.cu_region <parallel> {
// CHECK: sde.su_iterate (%c0) to (%c16) step (%c1)
// CHECK-SAME: reduction[#sde.reduction_kind<add>](%{{.+}} : memref<?xi32>)
// CHECK-SAME: classification(<reduction>) {
// CHECK: sde.cu_region <parallel> {
// CHECK: scf.for %{{.+}} = %c0 to %c4 step %c1
// CHECK: memref.load %arg0[%{{.+}}, %{{.+}}] : memref<16x4xi32>
// CHECK: memref.store %{{.+}}, %{{.+}}[%c0] : memref<?xi32>
// CHECK-NOT: linalg.generic
// CHECK: // -----// IR Dump After LoopInterchange

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  omp.declare_reduction @add_i32 : i32 init {
  ^bb0(%arg0: i32):
    %c0_i32 = arith.constant 0 : i32
    omp.yield(%c0_i32 : i32)
  } combiner {
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    omp.yield(%sum : i32)
  }

  func.func @main(%A: memref<16x4xi32>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %sum = memref.alloca() : memref<1xi32>
    %sum_cast = memref.cast %sum : memref<1xi32> to memref<?xi32>
    %c0_i32 = arith.constant 0 : i32
    memref.store %c0_i32, %sum[%c0] : memref<1xi32>
    omp.parallel {
      omp.wsloop reduction(@add_i32 %sum_cast -> %prv : memref<?xi32>) {
        omp.loop_nest (%i) : index = (%c0) to (%c16) step (%c1) {
          scf.for %j = %c0 to %c4 step %c1 {
            %val = memref.load %A[%i, %j] : memref<16x4xi32>
            %acc = memref.load %prv[%c0] : memref<?xi32>
            %next = arith.addi %acc, %val : i32
            memref.store %next, %prv[%c0] : memref<?xi32>
          }
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }
}
