// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from sde-planning --pipeline sde-planning --mlir-print-ir-after-all 2>&1 \
// RUN:   | awk '/IR Dump After ConvertOpenMPToSde/,/IR Dump After PatternAnalysis/' \
// RUN:   | %FileCheck %s --check-prefix=SINGLE
// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_multinode.cfg --start-from sde-planning --pipeline sde-planning --mlir-print-ir-after-all 2>&1 \
// RUN:   | awk '/IR Dump After ConvertOpenMPToSde/,/IR Dump After PatternAnalysis/' \
// RUN:   | %FileCheck %s --check-prefix=MULTI

// Unsupported triangular scatter writes are kept as host OpenMP for single-node
// benchmark rows until SDE/CODIR can express their sparse output ownership.
// Multinode runs still enter SDE so they remain true ARTS jobs.

// SINGLE-LABEL: // -----// IR Dump After ConvertOpenMPToSde (convert-openmp-to-sde) //----- //
// SINGLE: omp.parallel
// SINGLE: sde.keep_host_openmp
// SINGLE-NOT: sde.su_iterate

// MULTI-LABEL: // -----// IR Dump After ConvertOpenMPToSde (convert-openmp-to-sde) //----- //
// MULTI-NOT: sde.keep_host_openmp
// MULTI: sde.su_iterate

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i64:64-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<128x128xf64>, %C: memref<128x128xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %zero = arith.constant 0.0 : f64
    func.call @carts_benchmarks_start() : () -> ()

    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c0) to (%c128) step (%c1) {
          %next = arith.addi %i, %c1 : index
          scf.for %j = %next to %c128 step %c1 {
            %sum = scf.for %k = %c0 to %c128 step %c1 iter_args(%acc = %zero) -> (f64) {
              %a = memref.load %A[%i, %k] : memref<128x128xf64>
              %b = memref.load %A[%j, %k] : memref<128x128xf64>
              %mul = arith.mulf %a, %b : f64
              %next_acc = arith.addf %acc, %mul : f64
              scf.yield %next_acc : f64
            }
            memref.store %sum, %C[%i, %j] : memref<128x128xf64>
            memref.store %sum, %C[%j, %c0] : memref<128x128xf64>
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
