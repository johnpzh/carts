// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_64t.cfg --start-from sde-planning --pipeline sde-planning --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Verify that float add reductions stay on the tree path even when the
// single-node cost model would otherwise price integer atomics favorably.

// CHECK-LABEL: // -----// IR Dump After ReductionStrategy (reduction-strategy) //----- //
// CHECK: func.func @main
// CHECK: sde.cu_region <parallel> {
// CHECK: sde.su_iterate (%c0) to (%c128) step (%c1)
// CHECK-SAME: reduction{{\[}}#sde.reduction_kind<add>{{\]}}
// CHECK-SAME: reduction_strategy(<tree>)
// CHECK-NOT: reduction_strategy(<atomic>)

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  omp.declare_reduction @add_f32 : f32 init {
  ^bb0(%arg0: f32):
    %cst = arith.constant 0.000000e+00 : f32
    omp.yield(%cst : f32)
  } combiner {
  ^bb0(%lhs: f32, %rhs: f32):
    %sum = arith.addf %lhs, %rhs : f32
    omp.yield(%sum : f32)
  }

  func.func @main(%A: memref<128xf32>) {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %sum = memref.alloca() : memref<1xf32>
    %sum_cast = memref.cast %sum : memref<1xf32> to memref<?xf32>
    %zero = arith.constant 0.000000e+00 : f32
    memref.store %zero, %sum[%c0] : memref<1xf32>
    omp.parallel {
      omp.wsloop reduction(@add_f32 %sum_cast -> %prv : memref<?xf32>) {
        omp.loop_nest (%i) : index = (%c0) to (%c128) step (%c1) {
          %val = memref.load %A[%i] : memref<128xf32>
          %acc = memref.load %prv[%c0] : memref<?xf32>
          %next = arith.addf %acc, %val : f32
          memref.store %next, %prv[%c0] : memref<?xf32>
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }
}
