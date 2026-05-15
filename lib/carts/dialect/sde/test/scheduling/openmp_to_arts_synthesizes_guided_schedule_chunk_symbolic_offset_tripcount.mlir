// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_64t.cfg --start-from sde-planning --pipeline codir-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Verify that ChunkOpt synthesizes a symbolic chunk size for a
// guided loop whose trip count stays symbolic because both bounds are SSA
// values. The offset-bound path should reuse the same signed-safe span clamp
// while keeping the guided schedule at the SDE layer.

// CHECK-LABEL: // -----// IR Dump After ChunkOpt (chunk-opt) //----- //
// CHECK: func.func @main(%[[LB:.*]]: index, %[[UB:.*]]: index, %[[A:.*]]: memref<?xf64>, %[[B:.*]]: memref<?xf64>)
// The chunk computation uses the tiled trip count. Verify the
// balanced-chunk sequence exists and the su_iterate gets a chunk operand.
// CHECK: arith.subi %[[UB]], %[[LB]] : index
// CHECK: arith.cmpi slt
// CHECK: arith.select
// CHECK: arith.ceildivsi
// CHECK: arith.maxui
// CHECK: arith.ceildivui
// CHECK: arith.maxui
// CHECK: arith.minui
// CHECK: sde.su_iterate (%[[LB]]) to (%[[UB]]) step (%{{.*}}) schedule(<guided>, %{{.+}})

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%LB: index, %UB: index, %A: memref<?xf64>, %B: memref<?xf64>) {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 2.0 : f64
    omp.parallel {
      omp.wsloop schedule(guided, nonmonotonic) {
        omp.loop_nest (%i) : index = (%LB) to (%UB) step (%c1) {
          %v = memref.load %A[%i] : memref<?xf64>
          %r = arith.mulf %v, %cst : f64
          memref.store %r, %B[%i] : memref<?xf64>
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }
}
