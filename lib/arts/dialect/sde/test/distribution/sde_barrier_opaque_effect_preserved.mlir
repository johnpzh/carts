// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from openmp-to-arts --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Opaque memory effects inside either scheduling unit make barrier elimination
// conservative even when the modeled load/store roots look disjoint.

// CHECK-LABEL: // -----// IR Dump After BarrierElimination (barrier-elimination) //----- //
// CHECK: func.func @opaque_effect_preserves_barrier
// CHECK: arts_sde.su_iterate
// CHECK: arts_sde.su_barrier
// CHECK-NOT: barrierEliminated
// CHECK: arts_sde.su_iterate

// CHECK-LABEL: // -----// IR Dump After ConvertSdeToArts (convert-sde-to-arts) //----- //
// CHECK: func.func @opaque_effect_preserves_barrier
// CHECK: arts.edt <task>
// CHECK: arts.barrier
// CHECK: arts.edt <task>
// CHECK-NOT: arts_sde.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func private @opaque_touch(memref<128xf32>)

  func.func @opaque_effect_preserves_barrier(%A: memref<128xf32>, %B: memref<128xf32>, %C: memref<128xf32>, %D: memref<128xf32>) {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 1.000000e+00 : f32
    arts_sde.cu_region <parallel> scope(<local>) {
      arts_sde.su_iterate (%c0) to (%c128) step (%c1) classification(<elementwise>) {
      ^bb0(%i: index):
        %v = memref.load %A[%i] : memref<128xf32>
        memref.store %v, %B[%i] : memref<128xf32>
        arts_sde.yield
      }
      arts_sde.su_barrier
      arts_sde.su_iterate (%c0) to (%c128) step (%c1) classification(<elementwise>) {
      ^bb0(%i: index):
        func.call @opaque_touch(%C) : (memref<128xf32>) -> ()
        memref.store %cst, %D[%i] : memref<128xf32>
        arts_sde.yield
      }
      arts_sde.yield
    }
    return
  }
}
