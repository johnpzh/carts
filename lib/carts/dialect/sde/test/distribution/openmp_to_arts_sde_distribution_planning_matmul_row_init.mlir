// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_multinode.cfg --start-from sde-planning --pipeline codir-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=TILE
// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_multinode.cfg --start-from sde-planning --pipeline codir-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=SDE
// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_multinode.cfg --start-from sde-planning --pipeline codir-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=ARTS

// TILE-LABEL: // -----// IR Dump After Tiling (tiling) //----- //
// TILE: func.func @main
// TILE: sde.su_iterate (%c0) to (%c32) step (%{{.+}}) classification(<matmul>) {
// TILE: scf.for %[[I:[^ ]+]] = %{{.+}} to %{{.+}} step %c1 {
// TILE: scf.for %[[JT:[^ ]+]] = %c0 to %c32 step %{{.+}} {
// TILE: scf.for %[[J0:[^ ]+]] = %[[JT]] to %{{.+}} step %c1 {
// TILE: memref.store %{{.*}}, %{{.*}}[%[[I]], %[[J0]]] : memref<32x32xf32>
// TILE: scf.for %[[K:[^ ]+]] = %c0 to %c32 step %c1 {
// TILE: memref.store %{{.*}}, %{{.*}}[%[[I]], %{{.*}}] : memref<32x32xf32>
// TILE: } {
// TILE-SAME: iterationTopology = #sde.iteration_topology<owner_strip>
// TILE-SAME: logicalWorkerSlice = [1, 32]
// TILE-SAME: pattern = #sde.pattern<matmul>
// TILE-SAME: physicalBlockShape = [1, 32]
// TILE-SAME: physicalOwnerDims = [0]

// SDE-LABEL: // -----// IR Dump After DistributionPlanning (distribution-planning) //----- //
// SDE: func.func @main
// SDE: sde.su_iterate (%c0) to (%c32) step (%{{.+}}) classification(<matmul>) {
// SDE: } {
// SDE-SAME: iterationTopology = #sde.iteration_topology<owner_strip>
// SDE-SAME: logicalWorkerSlice = [1, 32]
// SDE-SAME: pattern = #sde.pattern<matmul>
// SDE-SAME: physicalBlockShape = [1, 32]
// SDE-SAME: physicalOwnerDims = [0]

// ARTS-LABEL: // -----// IR Dump After ConvertCodirToArts (convert-codir-to-arts) //----- //
// ARTS: func.func @main
// ARTS: arts.db_acquire{{.*}}{replicatedRead}
// ARTS: arts.edt <task>{{.*}}depPattern = #arts.dep_pattern<matmul>
// ARTS-SAME: planIterationTopology = #arts.plan_iteration_topology<owner_strip>
// ARTS-SAME: planLogicalWorkerSlice = [1, 32]
// ARTS-SAME: planOwnerDims = [0]
// ARTS-SAME: planPhysicalBlockShape = [1, 32]

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<32x32xf32>, %B: memref<32x32xf32>, %C: memref<32x32xf32>) {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.000000e+00 : f32
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c0) to (%c32) step (%c1) {
          scf.for %j = %c0 to %c32 step %c1 {
            memref.store %zero, %C[%i, %j] : memref<32x32xf32>
            scf.for %k = %c0 to %c32 step %c1 {
              %a = memref.load %A[%i, %k] : memref<32x32xf32>
              %b = memref.load %B[%k, %j] : memref<32x32xf32>
              %c = memref.load %C[%i, %j] : memref<32x32xf32>
              %prod = arith.mulf %a, %b : f32
              %sum = arith.addf %c, %prod : f32
              memref.store %sum, %C[%i, %j] : memref<32x32xf32>
            }
          }
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }
}
