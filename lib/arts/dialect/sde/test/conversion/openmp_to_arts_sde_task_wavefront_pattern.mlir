// RUN: %carts-compile %s --O3 --arts-config %arts_config --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=SDE
// RUN: %carts-compile %s --O3 --arts-config %arts_config --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=ARTS

// SDE owns task-dependency pattern classification. A two-dimensional task nest
// with one element out-dependency and two element in-dependencies on the same
// source is stamped as a wavefront before Core sees the task EDT.

// SDE-LABEL: // -----// IR Dump After ConvertOpenMPToSde (convert-openmp-to-sde) //----- //
// SDE: func.func @task_wavefront
// SDE: sde.mu_dep <write> %arg0[
// SDE-SAME: size[
// SDE: sde.mu_dep <read> %arg0[
// SDE-SAME: size[
// SDE: sde.mu_dep <read> %arg0[
// SDE-SAME: size[
// SDE: sde.cu_task deps(
// SDE: } {pattern = #sde.pattern<wavefront_2d>}

// ARTS-LABEL: // -----// IR Dump After ConvertSdeToArts (convert-sde-to-arts) //----- //
// ARTS: func.func @task_wavefront
// ARTS: arts.edt <task> <intranode>
// ARTS-SAME: depPattern = #arts.dep_pattern<wavefront_2d>
// ARTS-SAME: distribution_pattern = #arts.distribution_pattern<stencil>
// ARTS-NOT: sde.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @task_wavefront(%A: memref<?xi32>, %N: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32

    omp.parallel {
      scf.for %i = %c0 to %N step %c1 {
        scf.for %j = %c0 to %N step %c1 {
          %iN = arith.muli %i, %N : index
          %writeIdx = arith.addi %iN, %j : index
          %im1 = arith.subi %i, %c1 : index
          %im1N = arith.muli %im1, %N : index
          %topIdx = arith.addi %im1N, %j : index
          %jm1 = arith.subi %j, %c1 : index
          %leftIdx = arith.addi %iN, %jm1 : index

          %out = polygeist.subindex %A[%writeIdx] () : memref<?xi32> -> memref<?xi32>
          %top = polygeist.subindex %A[%topIdx] () : memref<?xi32> -> memref<?xi32>
          %left = polygeist.subindex %A[%leftIdx] () : memref<?xi32> -> memref<?xi32>
          omp.task depend(taskdependout -> %out : memref<?xi32>, taskdependin -> %top : memref<?xi32>, taskdependin -> %left : memref<?xi32>) {
            %t = memref.load %top[%c0] : memref<?xi32>
            %l = memref.load %left[%c0] : memref<?xi32>
            %sum = arith.addi %t, %l : i32
            %next = arith.addi %sum, %c1_i32 : i32
            memref.store %next, %out[%c0] : memref<?xi32>
            omp.terminator
          }
        }
      }
      omp.terminator
    }
    return
  }
}
