// RUN: %carts-compile %s --arts-config %arts_config --start-from concurrency --pipeline edt-distribution | %FileCheck %s

// The worker chunk loop is over the OpenMP-associated dimension, but the
// raised 3-D memref is contiguous in the last index. Core should keep the
// chosen DB/task partitioning and sink the task-local chunk loop inside the
// remaining scalar loops so the contiguous index is innermost.
//
// CHECK-LABEL: func.func @sink_task_loop_to_contiguous_dim
// CHECK: arts.edt <task>
// CHECK: scf.for %[[J:arg[0-9]+]] =
// CHECK: scf.for %[[I:arg[0-9]+]] =
// CHECK: scf.for %[[LOCAL:arg[0-9]+]] =
// CHECK: %[[GLOBAL:.+]] = arith.addi
// CHECK: memref.store %{{.*}}, %{{.*}}[%[[I]], %[[J]], %[[GLOBAL]]] : memref<?x?x?xf64>

module attributes {arts.runtime_total_nodes = 1 : i64, arts.runtime_total_workers = 4 : i64, dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @sink_task_loop_to_contiguous_dim() -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %route = arith.constant -1 : i32
    %zero = arith.constant 0.000000e+00 : f64

    %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>, <coarse>, <uniform>] route(%route : i32) sizes[%c1] elementType(f64) elementSizes[%c4, %c4, %c16] : (memref<?xi64>, memref<?xmemref<?x?x?xf64>>)
    %acq_guid, %acq_ptr = arts.db_acquire[<out>] (%guid : memref<?xi64>, %ptr : memref<?xmemref<?x?x?xf64>>) partitioning(<coarse>), indices[], offsets[%c0], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?x?x?xf64>>)

    arts.edt <parallel> <intranode> route(%route) (%acq_ptr) : memref<?xmemref<?x?x?xf64>> {
    ^bb0(%arg0: memref<?xmemref<?x?x?xf64>>):
      arts.for(%c0) to(%c16) step(%c1) {
      ^bb0(%k: index):
        scf.for %j = %c0 to %c4 step %c1 {
          scf.for %i = %c0 to %c4 step %c1 {
            %dst = arts.db_ref %arg0[%c0] : memref<?xmemref<?x?x?xf64>> -> memref<?x?x?xf64>
            memref.store %zero, %dst[%i, %j, %k] : memref<?x?x?xf64>
          }
        }
      }
      arts.db_release(%arg0) : memref<?xmemref<?x?x?xf64>>
    }

    arts.db_release(%acq_ptr) : memref<?xmemref<?x?x?xf64>>
    arts.db_free(%guid) : memref<?xi64>
    arts.db_free(%ptr) : memref<?xmemref<?x?x?xf64>>
    %ret = arith.constant 0 : i32
    return %ret : i32
  }
}
