// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from sde-planning --pipeline sde-planning --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Local libc allocator calls are accepted as scratch only when the allocation
// result is SSA-local to the analyzed body, used by the loop region, and freed
// without escaping.

// CHECK-LABEL: // -----// IR Dump After PatternAnalysis (sde-pattern-analysis) //----- //
// CHECK-LABEL: func.func @libc_allocator_local_scratch
// CHECK: sde.su_iterate (%c0) to (%c8) step (%c1) classification(<elementwise>) {
// CHECK: } {pattern = #sde.pattern<uniform>}

// CHECK-LABEL: func.func @libc_allocator_escape_rejected
// CHECK: sde.su_iterate (%c0) to (%c8) step (%c1) {
// CHECK-NOT: classification
// CHECK: memref.store %{{.*}}, %{{.*}}[] : memref<memref<?xf64>>

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func private @malloc(index) -> memref<?xf64>
  func.func private @free(memref<?xf64>)

  func.func @libc_allocator_local_scratch(%A: memref<8x4xf64>, %B: memref<8x4xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    sde.cu_region <parallel> {
      sde.su_iterate (%c0) to (%c8) step (%c1) {
      ^bb0(%i: index):
        %scratch = func.call @malloc(%c4) : (index) -> memref<?xf64>
        scf.for %j = %c0 to %c4 step %c1 {
          %a = memref.load %A[%i, %j] : memref<8x4xf64>
          memref.store %a, %scratch[%j] : memref<?xf64>
          %tmp = memref.load %scratch[%j] : memref<?xf64>
          memref.store %tmp, %B[%i, %j] : memref<8x4xf64>
        }
        func.call @free(%scratch) : (memref<?xf64>) -> ()
        sde.yield
      }
      sde.yield
    }
    return
  }

  func.func @libc_allocator_escape_rejected(%A: memref<8x4xf64>, %B: memref<8x4xf64>, %slot: memref<memref<?xf64>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    sde.cu_region <parallel> {
      sde.su_iterate (%c0) to (%c8) step (%c1) {
      ^bb0(%i: index):
        %scratch = func.call @malloc(%c4) : (index) -> memref<?xf64>
        memref.store %scratch, %slot[] : memref<memref<?xf64>>
        scf.for %j = %c0 to %c4 step %c1 {
          %a = memref.load %A[%i, %j] : memref<8x4xf64>
          memref.store %a, %scratch[%j] : memref<?xf64>
          memref.store %a, %B[%i, %j] : memref<8x4xf64>
        }
        func.call @free(%scratch) : (memref<?xf64>) -> ()
        sde.yield
      }
      sde.yield
    }
    return
  }
}
