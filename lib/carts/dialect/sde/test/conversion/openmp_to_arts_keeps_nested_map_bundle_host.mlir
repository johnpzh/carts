// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from sde-planning --pipeline sde-planning --mlir-print-ir-after-all 2>&1 \
// RUN:   | awk '/IR Dump After ConvertOpenMPToSde/,/IR Dump After PatternAnalysis/' \
// RUN:   | %FileCheck %s --check-prefix=SINGLE
// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_multinode.cfg --start-from sde-planning --pipeline sde-planning --mlir-print-ir-after-all 2>&1 \
// RUN:   | awk '/IR Dump After ConvertOpenMPToSde/,/IR Dump After PatternAnalysis/' \
// RUN:   | %FileCheck %s --check-prefix=MULTI

// Single-node host-island policy must not change multinode lowering.

// SINGLE-LABEL: // -----// IR Dump After ConvertOpenMPToSde (convert-openmp-to-sde) //----- //
// SINGLE: omp.parallel
// SINGLE: sde.keep_host_openmp
// SINGLE-NOT: sde.su_iterate

// MULTI-LABEL: // -----// IR Dump After ConvertOpenMPToSde (convert-openmp-to-sde) //----- //
// MULTI-NOT: sde.keep_host_openmp
// MULTI: sde.su_iterate

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i64:64-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%input: memref<8x16x64xf32>, %tmp: memref<8x16x64xf32>, %output: memref<8x16x64xf32>, %global: memref<8x16xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %zero = arith.constant 0.0 : f32
    func.call @carts_benchmarks_start() : () -> ()

    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%b) : index = (%c0) to (%c8) step (%c1) {
          scf.for %c = %c0 to %c16 step %c1 {
            scf.for %i = %c0 to %c16 step %c1 {
              scf.for %j = %c0 to %c16 step %c1 {
                %ii = arith.muli %i, %c2 : index
                %idx = arith.addi %ii, %j : index
                %v = memref.load %input[%b, %c, %idx] : memref<8x16x64xf32>
                memref.store %v, %tmp[%b, %c, %idx] : memref<8x16x64xf32>
              }
            }
          }
          omp.yield
        }
      }
      omp.terminator
    }

    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%b) : index = (%c0) to (%c8) step (%c1) {
          scf.for %c = %c0 to %c16 step %c1 {
            scf.for %i = %c0 to %c16 step %c1 {
              scf.for %j = %c0 to %c16 step %c1 {
                %idx = arith.addi %i, %j : index
                %v = memref.load %tmp[%b, %c, %idx] : memref<8x16x64xf32>
                %r = arith.addf %v, %zero : f32
                memref.store %r, %output[%b, %c, %idx] : memref<8x16x64xf32>
              }
            }
          }
          omp.yield
        }
      }
      omp.terminator
    }

    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%b) : index = (%c0) to (%c8) step (%c1) {
          scf.for %c = %c0 to %c16 step %c1 {
            scf.for %i = %c0 to %c16 step %c1 {
              scf.for %j = %c0 to %c16 step %c1 {
                %idx = arith.addi %i, %j : index
                %v = memref.load %output[%b, %c, %idx] : memref<8x16x64xf32>
                memref.store %v, %global[%b, %c] : memref<8x16xf32>
              }
            }
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
