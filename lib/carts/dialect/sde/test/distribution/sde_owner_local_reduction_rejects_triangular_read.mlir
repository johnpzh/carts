// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_multinode_64x64.cfg --start-from sde-planning --pipeline sde-planning --mlir-print-ir-after-all 2>&1 | %FileCheck %s
// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_64t.cfg --start-from sde-planning --emit-llvm | %FileCheck %s --check-prefix=LLVM

// Correlation-style triangular self-Gram kernels can look matmul-like after
// structured analysis, but the same external input root supplies both logical
// sides of the multiply. CARTS currently has one physical dependency view per
// root, so this must not receive an owner-slice plan; a compute-block view for
// data[i, *] would not cover data[j, *] across the upper triangle.

// CHECK-LABEL: // -----// IR Dump After DistributionPlanning (distribution-planning) //----- //
// CHECK-LABEL: func.func @triangular_correlation_read_not_owner_local
// CHECK: sde.su_iterate
// CHECK-SAME: classification(<matmul>)
// CHECK-NOT: physicalOwnerDims
// CHECK-LABEL: // -----// IR Dump After IterationSpaceDecomposition

// LLVM-LABEL: define void @__arts_edt_1
// LLVM-NOT: sub i64
// LLVM: ret void

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @triangular_correlation_read_not_owner_local(%data: memref<32x32xf32>, %corr: memref<32x32xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %one = arith.constant 1.0 : f32
    %zero = arith.constant 0.0 : f32
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c0) to (%c32) step (%c1) {
          memref.store %one, %corr[%i, %i] : memref<32x32xf32>
          %ip1 = arith.addi %i, %c1 : index
          scf.for %j = %ip1 to %c32 step %c1 {
            %sum = scf.for %k = %c0 to %c32 step %c1 iter_args(%acc = %zero) -> (f32) {
              %a = memref.load %data[%i, %k] : memref<32x32xf32>
              %b = memref.load %data[%j, %k] : memref<32x32xf32>
              %prod = arith.mulf %a, %b : f32
              %next = arith.addf %acc, %prod : f32
              scf.yield %next : f32
            }
            memref.store %sum, %corr[%i, %j] : memref<32x32xf32>
            memref.store %sum, %corr[%j, %i] : memref<32x32xf32>
          }
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }
}
