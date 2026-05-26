// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from sde-planning --pipeline sde-planning --mlir-print-ir-after-all 2>&1 \
// RUN:   | awk '/IR Dump After ConvertOpenMPToSde/,/IR Dump After PatternAnalysis/' \
// RUN:   | %FileCheck %s --check-prefix=SINGLE
// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_multinode.cfg --start-from sde-planning --pipeline sde-planning --mlir-print-ir-after-all 2>&1 \
// RUN:   | awk '/IR Dump After ConvertOpenMPToSde/,/IR Dump After PatternAnalysis/' \
// RUN:   | %FileCheck %s --check-prefix=MULTI

// Dense two-stage matrix-vector bundles use the single-node host policy.

// SINGLE-LABEL: // -----// IR Dump After ConvertOpenMPToSde (convert-openmp-to-sde) //----- //
// SINGLE: omp.parallel
// SINGLE: sde.keep_host_openmp
// SINGLE-NOT: sde.su_iterate

// MULTI-LABEL: // -----// IR Dump After ConvertOpenMPToSde (convert-openmp-to-sde) //----- //
// MULTI-NOT: sde.keep_host_openmp
// MULTI: sde.su_iterate

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i64:64-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%a: memref<8x16xf64>, %x: memref<16xf64>, %tmp: memref<8xf64>, %y: memref<16xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %zero = arith.constant 0.0 : f64
    func.call @carts_benchmarks_start() : () -> ()

    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c0) to (%c8) step (%c1) {
          memref.store %zero, %tmp[%i] : memref<8xf64>
          scf.for %j = %c0 to %c16 step %c1 {
            %acc = memref.load %tmp[%i] : memref<8xf64>
            %av = memref.load %a[%i, %j] : memref<8x16xf64>
            %xv = memref.load %x[%j] : memref<16xf64>
            %prod = arith.mulf %av, %xv : f64
            %sum = arith.addf %acc, %prod : f64
            memref.store %sum, %tmp[%i] : memref<8xf64>
          }
          omp.yield
        }
      }
      omp.terminator
    }

    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%j) : index = (%c0) to (%c16) step (%c1) {
          memref.store %zero, %y[%j] : memref<16xf64>
          scf.for %i = %c0 to %c8 step %c1 {
            %acc = memref.load %y[%j] : memref<16xf64>
            %av = memref.load %a[%i, %j] : memref<8x16xf64>
            %tv = memref.load %tmp[%i] : memref<8xf64>
            %prod = arith.mulf %av, %tv : f64
            %sum = arith.addf %acc, %prod : f64
            memref.store %sum, %y[%j] : memref<16xf64>
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
