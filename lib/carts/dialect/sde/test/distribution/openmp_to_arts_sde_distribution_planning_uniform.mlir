// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from sde-planning --pipeline codir-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=SDE
// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from sde-planning --pipeline codir-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=ARTS

// Verify that SDE authors a blocked distribution advisory for a local
// elementwise loop, and that boundary materialization consumes that advisory
// without leaving `sde.su_distribute` in the IR. It should also author the
// physical storage layout for independent output-vector loops whose bodies
// contain loop-local scratch/reduction work and therefore are not classified as
// pure elementwise linalg kernels.

// SDE-LABEL: // -----// IR Dump After DistributionPlanning (distribution-planning) //----- //
// SDE: func.func @main
// SDE: sde.cu_region <parallel> {
// SDE: sde.su_distribute <blocked> {
// SDE: sde.su_iterate (%c0) to (%c128) step (%{{.+}}) classification(<elementwise>) {

// SDE-LABEL: func.func @sample_map_with_scratch
// SDE: sde.su_distribute <blocked> {
// SDE: sde.su_iterate
// SDE: } {
// SDE-SAME: iterationTopology = #sde.iteration_topology<owner_strip>
// SDE-SAME: physicalBlockShape = [8]
// SDE-SAME: physicalOwnerDims = [0]
// SDE-NOT: {{plan[A-Z]}}
// SDE-LABEL: // -----// IR Dump After IterationSpaceDecomposition

// ARTS-LABEL: // -----// IR Dump After ConvertCodirToArts (convert-codir-to-arts) //----- //
// ARTS: func.func @main
// ARTS: arts.edt <task>
// ARTS: arts.edt <task>{{.*}}arts.pattern_revision = 1 : i64{{.*}}depPattern = #arts.dep_pattern<uniform>{{.*}}distribution_kind = #arts.distribution_kind<block>{{.*}}distribution_pattern = #arts.distribution_pattern<uniform>{{.*}}distribution_version = 1 : i32
// ARTS-NOT: sde.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<128xf64>, %B: memref<128xf64>) {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 2.0 : f64
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c0) to (%c128) step (%c1) {
          %v = memref.load %A[%i] : memref<128xf64>
          %r = arith.mulf %v, %cst : f64
          memref.store %r, %B[%i] : memref<128xf64>
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }

  func.func @sample_map_with_scratch() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f64
    %out = memref.alloc() : memref<64xf64>
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%s) : index = (%c0) to (%c64) step (%c1) {
          %scratch = memref.alloc() : memref<4xf64>
          memref.store %cst, %scratch[%c0] : memref<4xf64>
          scf.for %i = %c0 to %c4 step %c1 {
            %v = memref.load %scratch[%c0] : memref<4xf64>
            %next = arith.addf %v, %cst : f64
            memref.store %next, %scratch[%c0] : memref<4xf64>
          }
          %sum = memref.load %scratch[%c0] : memref<4xf64>
          memref.store %sum, %out[%s] : memref<64xf64>
          memref.dealloc %scratch : memref<4xf64>
          omp.yield
        }
      }
      omp.terminator
    }
    %check = memref.load %out[%c0] : memref<64xf64>
    func.call @use(%check) : (f64) -> ()
    memref.dealloc %out : memref<64xf64>
    return %c0_i32 : i32
  }

  func.func private @use(f64) attributes {llvm.linkage = #llvm.linkage<external>}
}
