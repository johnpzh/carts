// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from sde-planning --pipeline codir-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Symmetric self-Gram loops compute one dot product and scatter it to both
// triangles. LoopInterchange splits the mirrored lower-triangle write into a
// separate coarse mirror phase so the heavy dot-product phase owns rows of the
// output.

// CHECK-LABEL: // -----// IR Dump After LoopInterchange (loop-interchange) //----- //
// CHECK: sde.su_iterate
// CHECK-SAME: classification(<matmul>)
// CHECK: memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<16x16xf64>
// CHECK: physicalBlockShape = [1, 16]
// CHECK-SAME: physicalOwnerDims = [0]
// CHECK: sde.su_iterate
// CHECK: memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<16x16xf64>
// CHECK: memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<16x16xf64>

// CHECK-LABEL: // -----// IR Dump After ConvertSdeToCodir (convert-sde-to-codir) //----- //
// CHECK: codir.codelet deps(%{{.*}}, %{{.*}} : memref<16x16xf64>, memref<16x16xf64>)
// CHECK-SAME: dep_storage_views = [#codir.storage_view<host_whole>, #codir.storage_view<compute_block>]
// CHECK-SAME: pattern = #codir.pattern<matmul>
// CHECK: codir.codelet deps(%{{.*}} : memref<16x16xf64>)
// CHECK-SAME: dep_storage_views = [#codir.storage_view<host_whole>]

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<16x16xf64>, %C: memref<16x16xf64>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %one = arith.constant 1.0 : f64
    %zero = arith.constant 0.0 : f64
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c0) to (%c16) step (%c1) {
          memref.store %one, %C[%i, %i] : memref<16x16xf64>
          %next = arith.addi %i, %c1 : index
          scf.for %j = %next to %c16 step %c1 {
            %sum = scf.for %k = %c0 to %c16 step %c1 iter_args(%acc = %zero) -> (f64) {
              %a = memref.load %A[%i, %k] : memref<16x16xf64>
              %b = memref.load %A[%j, %k] : memref<16x16xf64>
              %prod = arith.mulf %a, %b : f64
              %next_acc = arith.addf %acc, %prod : f64
              scf.yield %next_acc : f64
            }
            memref.store %sum, %C[%i, %j] : memref<16x16xf64>
            memref.store %sum, %C[%j, %i] : memref<16x16xf64>
          }
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }
}
