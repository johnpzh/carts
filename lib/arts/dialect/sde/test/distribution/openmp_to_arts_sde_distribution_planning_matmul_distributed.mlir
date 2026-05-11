// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_multinode.cfg --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=SDE
// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_multinode.cfg --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=ARTS

// Verify that distributed-scope matmul loops with enough work now receive
// blocked distribution planning and preserve the matmul contract across the
// SDE-to-ARTS boundary.

// SDE-LABEL: // -----// IR Dump After DistributionPlanning (distribution-planning) //----- //
// SDE: func.func @main
// SDE: arts_sde.cu_region <parallel> scope(<distributed>) {
// SDE: arts_sde.su_distribute <blocked> {
// SDE: arts_sde.su_iterate (%c0) to (%c32) step (%{{.+}}) classification(<matmul>) {

// ARTS-LABEL: // -----// IR Dump After ConvertSdeToArts (convert-sde-to-arts) //----- //
// ARTS: func.func @main
// ARTS: arts.edt <parallel> <internode> route(%{{.*}}) attributes {
// ARTS: arts.for(%c0) to(%c32) step(%{{.+}}) {
// ARTS: } {arts.pattern_revision = 1 : i64
// ARTS-SAME: depPattern = #arts.dep_pattern<matmul>
// ARTS-SAME: distribution_pattern = #arts.distribution_pattern<matmul>

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<32x32xf32>, %B: memref<32x32xf32>, %C: memref<32x32xf32>) {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c0) to (%c32) step (%c1) {
          scf.for %j = %c0 to %c32 step %c1 {
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
