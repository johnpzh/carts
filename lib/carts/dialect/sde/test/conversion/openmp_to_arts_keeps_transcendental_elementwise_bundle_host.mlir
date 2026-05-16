// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from sde-planning --pipeline sde-planning --mlir-print-ir-after-all 2>&1 \
// RUN:   | awk '/IR Dump After ConvertOpenMPToSde/,/IR Dump After PatternAnalysis/' \
// RUN:   | %FileCheck %s

// A bundle of independent one-dimensional floating-point maps with repeated
// transcendental work is kept as host OpenMP. The fused SDE elementwise path
// would place cheap maps in the same hot loop as scalar libm work and lose the
// source OpenMP schedule shape.

// CHECK-LABEL: // -----// IR Dump After ConvertOpenMPToSde (convert-openmp-to-sde) //----- //
// CHECK: omp.parallel
// CHECK: sde.keep_host_openmp
// CHECK-NOT: sde.su_iterate
// CHECK: func.func private @tanhf

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i64:64-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<1024xf32>, %B: memref<1024xf32>, %C: memref<1024xf32>, %D: memref<1024xf32>, %E: memref<1024xf32>) {
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32
    %one = arith.constant 1.0 : f32
    %scale = arith.constant 1.702 : f32
    func.call @carts_benchmarks_start() : () -> ()

    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c0) to (%c1024) step (%c1) {
          %v = memref.load %A[%i] : memref<1024xf32>
          %cmp = arith.cmpf ogt, %v, %zero : f32
          %out = arith.select %cmp, %v, %zero : f32
          memref.store %out, %B[%i] : memref<1024xf32>
          omp.yield
        }
      }
      omp.terminator
    }

    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c0) to (%c1024) step (%c1) {
          %v = memref.load %A[%i] : memref<1024xf32>
          %mul = arith.mulf %v, %scale : f32
          memref.store %mul, %C[%i] : memref<1024xf32>
          omp.yield
        }
      }
      omp.terminator
    }

    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c0) to (%c1024) step (%c1) {
          %v = memref.load %A[%i] : memref<1024xf32>
          %neg = arith.negf %v : f32
          %exp = math.exp %neg : f32
          %den = arith.addf %exp, %one : f32
          %out = arith.divf %one, %den : f32
          memref.store %out, %D[%i] : memref<1024xf32>
          omp.yield
        }
      }
      omp.terminator
    }

    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c0) to (%c1024) step (%c1) {
          %v = memref.load %A[%i] : memref<1024xf32>
          %out = func.call @tanhf(%v) : (f32) -> f32
          memref.store %out, %E[%i] : memref<1024xf32>
          omp.yield
        }
      }
      omp.terminator
    }

    return
  }

  func.func private @tanhf(f32) -> f32
  func.func private @carts_benchmarks_start()
}
