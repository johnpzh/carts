// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from openmp-to-arts --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Verify that ConvertSdeToArts materializes a cyclic SDE advisory as the
// corresponding ARTS block-cyclic distribution kind.

// CHECK-LABEL: // -----// IR Dump After ConvertSdeToArts (convert-sde-to-arts) //----- //
// CHECK: func.func @main
// CHECK: arts.epoch attributes {
// CHECK-SAME: {{.*}}depPattern = #arts.dep_pattern<uniform>{{.*}}distribution_kind = #arts.distribution_kind<block_cyclic>{{.*}}distribution_pattern = #arts.distribution_pattern<uniform>{{.*}}distribution_version = 1 : i32
// CHECK: arts.edt <task>
// CHECK-SAME: {{.*}}depPattern = #arts.dep_pattern<uniform>{{.*}}distribution_kind = #arts.distribution_kind<block_cyclic>{{.*}}distribution_pattern = #arts.distribution_pattern<uniform>{{.*}}distribution_version = 1 : i32
// CHECK-SAME: no_verify = #arts.no_verify
// CHECK-NOT: arts_sde.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<16xf64>, %B: memref<16xf64>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    arts_sde.cu_region <parallel> scope(<local>) {
      arts_sde.su_distribute <cyclic> {
        arts_sde.su_iterate (%c0) to (%c16) step (%c1) classification(<elementwise>) {
        ^bb0(%i: index):
          %v = memref.load %A[%i] : memref<16xf64>
          memref.store %v, %B[%i] : memref<16xf64>
          arts_sde.yield
        }
      }
      arts_sde.yield
    }
    return
  }
}
