// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from sde-planning --pipeline codir-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// CHECK-LABEL: // -----// IR Dump After ConvertOpenMPToSde (convert-openmp-to-sde) //----- //
// CHECK: func.func @main
// CHECK: sde.su_iterate
// CHECK-NOT: sde.keep_host_openmp
// CHECK-LABEL: // -----// IR Dump After PatternAnalysis (sde-pattern-analysis) //----- //
// CHECK: sde.su_iterate (%c0) to (%c32) step (%{{.+}}) classification(<matmul>)
// CHECK-LABEL: // -----// IR Dump After ConvertCodirToArts (convert-codir-to-arts) //----- //
// CHECK: arts.edt <task>{{.*}}depPattern = #arts.dep_pattern<matmul>
// CHECK-SAME: distribution_pattern = #arts.distribution_pattern<matmul>

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<32x32xf32>, %B: memref<32x32xf32>, %C: memref<32x32xf32>) {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.000000e+00 : f32
    func.call @carts_benchmarks_start() : () -> ()

    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c0) to (%c32) step (%c1) {
          %scratch = memref.alloca() : memref<f32>
          memref.store %zero, %scratch[] : memref<f32>
          scf.for %j = %c0 to %c32 step %c1 {
            memref.store %zero, %scratch[] : memref<f32>
            scf.for %k = %c0 to %c32 step %c1 {
              %a = memref.load %A[%i, %k] : memref<32x32xf32>
              %b = memref.load %B[%k, %j] : memref<32x32xf32>
              %prod = arith.mulf %a, %b : f32
              %old = memref.load %scratch[] : memref<f32>
              %sum = arith.addf %old, %prod : f32
              memref.store %sum, %scratch[] : memref<f32>
            }
            %acc = memref.load %scratch[] : memref<f32>
            memref.store %acc, %C[%i, %j] : memref<32x32xf32>
          }
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }

  func.func private @carts_benchmarks_start()
}
