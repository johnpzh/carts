// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from sde-planning --pipeline sde-planning --mlir-print-ir-after-all 2>&1 \
// RUN:   | awk '/IR Dump After ConvertOpenMPToSde/,/IR Dump After PatternAnalysis/' \
// RUN:   | %FileCheck %s --check-prefix=SINGLE
// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_multinode.cfg --start-from sde-planning --pipeline sde-planning --mlir-print-ir-after-all 2>&1 \
// RUN:   | awk '/IR Dump After ConvertOpenMPToSde/,/IR Dump After PatternAnalysis/' \
// RUN:   | %FileCheck %s --check-prefix=MULTI

// A repeated bundle of one-dimensional floating-point streaming maps is kept as
// host OpenMP for single-node benchmark runs. Multinode runs must enter
// SDE/CODIR/ARTS so benchmark Slurm artifacts are true distributed ARTS jobs.

// SINGLE-LABEL: // -----// IR Dump After ConvertOpenMPToSde (convert-openmp-to-sde) //----- //
// SINGLE: omp.parallel
// SINGLE: sde.keep_host_openmp
// SINGLE-NOT: sde.su_iterate

// MULTI-LABEL: // -----// IR Dump After ConvertOpenMPToSde (convert-openmp-to-sde) //----- //
// MULTI-NOT: sde.keep_host_openmp
// MULTI: sde.su_iterate

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i64:64-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<1024xf64>, %B: memref<1024xf64>, %C: memref<1024xf64>) {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %scalar = arith.constant 3.0 : f64
    func.call @carts_benchmarks_start() : () -> ()

    scf.for %k = %c0 to %c10 step %c1 {
      omp.parallel {
        omp.wsloop {
          omp.loop_nest (%i) : index = (%c0) to (%c1024) step (%c1) {
            %v = memref.load %A[%i] : memref<1024xf64>
            memref.store %v, %C[%i] : memref<1024xf64>
            omp.yield
          }
        }
        omp.terminator
      }

      omp.parallel {
        omp.wsloop {
          omp.loop_nest (%i) : index = (%c0) to (%c1024) step (%c1) {
            %v = memref.load %C[%i] : memref<1024xf64>
            %r = arith.mulf %scalar, %v : f64
            memref.store %r, %B[%i] : memref<1024xf64>
            omp.yield
          }
        }
        omp.terminator
      }

      omp.parallel {
        omp.wsloop {
          omp.loop_nest (%i) : index = (%c0) to (%c1024) step (%c1) {
            %a = memref.load %A[%i] : memref<1024xf64>
            %b = memref.load %B[%i] : memref<1024xf64>
            %r = arith.addf %a, %b : f64
            memref.store %r, %C[%i] : memref<1024xf64>
            omp.yield
          }
        }
        omp.terminator
      }

      omp.parallel {
        omp.wsloop {
          omp.loop_nest (%i) : index = (%c0) to (%c1024) step (%c1) {
            %b = memref.load %B[%i] : memref<1024xf64>
            %c = memref.load %C[%i] : memref<1024xf64>
            %mul = arith.mulf %scalar, %c : f64
            %r = arith.addf %b, %mul : f64
            memref.store %r, %A[%i] : memref<1024xf64>
            omp.yield
          }
        }
        omp.terminator
      }
    }

    return
  }

  func.func private @carts_benchmarks_start()
}
