// RUN: %carts-compile %s --O3 --arts-config %arts_config --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Verify the before/after for SDE boundary ownership:
// ConvertOpenMPToSde must not copy generic ARTS bookkeeping metadata onto new
// SDE loop ops.

// CHECK-LABEL: // -----// IR Dump After ConvertOpenMPToSde (convert-openmp-to-sde) //----- //
// CHECK: func.func @wsloop_clean
// CHECK: sde.cu_region <parallel> {
// CHECK: sde.su_iterate (%c0) to (%c8) step (%c1) {
// CHECK: func.func @taskloop_clean
// CHECK: sde.cu_region <parallel> {
// CHECK: sde.su_iterate (%c0) to (%c8) step (%c1) {
// CHECK: func.func @scf_parallel_clean
// CHECK: sde.cu_region <parallel> {
// CHECK: sde.su_iterate (%c0) to (%c8) step (%c1) {
// CHECK-NOT: 7001
// CHECK-NOT: 7101
// CHECK-NOT: 7002
// CHECK-NOT: 7102
// CHECK-NOT: 7003
// CHECK-NOT: 7103
// CHECK-NOT: arts.metadata_provenance
// CHECK-NOT: arts.partition_hint
// CHECK-NOT: partition_mode

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @wsloop_clean(%A: memref<8xf64>, %B: memref<8xf64>) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c0) to (%c8) step (%c1) {
          %v = memref.load %A[%i] : memref<8xf64>
          memref.store %v, %B[%i] : memref<8xf64>
          omp.yield
        }
      } {arts.id = 7001 : i64, arts.metadata_origin_id = 7101 : i64, arts.metadata_provenance = "transferred"}
      omp.terminator
    }
    return
  }

  func.func @taskloop_clean(%A: memref<8xf64>, %B: memref<8xf64>) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    omp.parallel {
      omp.taskloop {
        omp.loop_nest (%i) : index = (%c0) to (%c8) step (%c1) {
          %v = memref.load %A[%i] : memref<8xf64>
          memref.store %v, %B[%i] : memref<8xf64>
          omp.yield
        }
      } {arts.id = 7002 : i64, arts.metadata_origin_id = 7102 : i64, arts.metadata_provenance = "transferred"}
      omp.terminator
    }
    return
  }

  func.func @scf_parallel_clean(%A: memref<8xf64>, %B: memref<8xf64>) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    scf.parallel (%i) = (%c0) to (%c8) step (%c1) {
      %v = memref.load %A[%i] : memref<8xf64>
      memref.store %v, %B[%i] : memref<8xf64>
      scf.reduce
    } {arts.id = 7003 : i64, arts.metadata_origin_id = 7103 : i64, arts.metadata_provenance = "transferred"}
    return
  }
}
