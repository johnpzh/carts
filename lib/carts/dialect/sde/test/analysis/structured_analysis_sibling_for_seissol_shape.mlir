// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from sde-planning --pipeline sde-planning --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Sibling scf.for loops in one scheduling-unit body are a structured region
// only when non-loop boundary operations are effect-free or proven local
// scratch. This matches SeisSol-style q-then-b loop bodies without choosing one
// sibling as the single innermost loop.

// CHECK-LABEL: // -----// IR Dump After PatternAnalysis (sde-pattern-analysis) //----- //
// CHECK-LABEL: func.func @sibling_for_seissol_shape
// CHECK: sde.su_iterate (%c0) to (%c8) step (%c1) classification(<elementwise>) {
// CHECK: } {pattern = #sde.pattern<uniform>}

// CHECK-LABEL: func.func @sibling_for_opaque_boundary_rejected
// CHECK: sde.su_iterate (%c0) to (%c8) step (%c1) {
// CHECK-NOT: classification
// CHECK: func.call @opaque_boundary

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func private @opaque_boundary()

  func.func @sibling_for_seissol_shape(%A: memref<8x4xf64>, %B: memref<8x4xf64>, %C: memref<8x8xf64>, %D: memref<8x8xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    sde.cu_region <parallel> {
      sde.su_iterate (%c0) to (%c8) step (%c1) {
      ^bb0(%i: index):
        scf.for %q = %c0 to %c4 step %c1 {
          %a = memref.load %A[%i, %q] : memref<8x4xf64>
          memref.store %a, %B[%i, %q] : memref<8x4xf64>
        }
        scf.for %b = %c0 to %c8 step %c1 {
          %c = memref.load %C[%i, %b] : memref<8x8xf64>
          memref.store %c, %D[%i, %b] : memref<8x8xf64>
        }
        sde.yield
      }
      sde.yield
    }
    return
  }

  func.func @sibling_for_opaque_boundary_rejected(%A: memref<8x4xf64>, %B: memref<8x4xf64>, %C: memref<8x8xf64>, %D: memref<8x8xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    sde.cu_region <parallel> {
      sde.su_iterate (%c0) to (%c8) step (%c1) {
      ^bb0(%i: index):
        scf.for %q = %c0 to %c4 step %c1 {
          %a = memref.load %A[%i, %q] : memref<8x4xf64>
          memref.store %a, %B[%i, %q] : memref<8x4xf64>
        }
        func.call @opaque_boundary() : () -> ()
        scf.for %b = %c0 to %c8 step %c1 {
          %c = memref.load %C[%i, %b] : memref<8x8xf64>
          memref.store %c, %D[%i, %b] : memref<8x8xf64>
        }
        sde.yield
      }
      sde.yield
    }
    return
  }
}
