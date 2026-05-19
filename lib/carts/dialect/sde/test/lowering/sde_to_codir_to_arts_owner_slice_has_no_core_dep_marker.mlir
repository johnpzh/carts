// RUN: %carts-compile %s --O3 --arts-config %arts_config \
// RUN:   --start-from sde-planning --pipeline codir-to-arts \
// RUN:   | %FileCheck %s --implicit-check-not=sde. --implicit-check-not=codir.codelet
// RUN: %carts-compile %s --O3 --arts-config %arts_config \
// RUN:   --start-from sde-planning --emit-llvm \
// RUN:   | %FileCheck %s --check-prefix=LLVM

// Owner-slice scheduling units become canonical MU tokens/codelets before ARTS,
// preserving block partition metadata through CODIR-to-ARTS materialization.

// CHECK-LABEL: func.func @owner_slice_no_core_dep_marker
// CHECK: arts.db_alloc{{.*}}<coarse>
// CHECK-SAME: elementSizes[%c128{{(_[0-9]+)?}}, %c64{{(_[0-9]+)?}}]
// CHECK: arts.db_alloc{{.*}}<block>
// CHECK-SAME: elementSizes[%c16{{(_[0-9]+)?}}, %c64{{(_[0-9]+)?}}]
// CHECK-SAME: arts.storage_bridge = "host_whole_to_compute_block"
// CHECK-SAME: planPhysicalBlockShape = [16, 64]
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.store
// CHECK: arts.db_acquire[<in>]
// CHECK-SAME: partitioning(<block>)
// CHECK: arts.db_acquire[<inout>]
// CHECK-SAME: partitioning(<block>)
// CHECK: arts.edt <task> <intranode>
// CHECK: arts.db_ref
// CHECK: %[[LOCAL_IN:.*]] = arith.subi %arg{{[0-9]+}}, %arg{{[0-9]+}} : index
// CHECK-NEXT: memref.load {{.*}}[%[[LOCAL_IN]],
// CHECK: memref.store
// CHECK: arts.barrier
// CHECK-SAME: barrierReason = #arts.barrier_reason<required_memory>
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.store

// LLVM: define void @owner_slice_no_core_dep_marker

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @owner_slice_no_core_dep_marker(%A: memref<128x64xf32>, %C: memref<128x64xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index

    sde.cu_region <parallel> {
      sde.su_iterate (%c0) to (%c128) step (%c1) classification(<elementwise>) {
      ^bb0(%i: index):
        scf.for %j = %c0 to %c64 step %c1 {
          %v = memref.load %A[%i, %j] : memref<128x64xf32>
          %old = memref.load %C[%i, %j] : memref<128x64xf32>
          %next = arith.addf %old, %v : f32
          memref.store %next, %C[%i, %j] : memref<128x64xf32>
        }
        sde.yield
      } {iterationTopology = #sde.iteration_topology<owner_strip>, logicalWorkerSlice = [16, 64], pattern = #sde.pattern<uniform>, physicalBlockShape = [16, 64], physicalOwnerDims = [0]}
      sde.yield
    }

    return
  }
}
