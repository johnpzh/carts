// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_multinode_64x64.cfg \
// RUN:   --start-from sde-planning --pipeline sde-to-codir | %FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @owner_local_reduction_uses_single_channel_grain(%A: memref<1920x16xf32>, %y: memref<1920xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c1920 = arith.constant 1920 : index
    %zero = arith.constant 0.0 : f32
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c0) to (%c1920) step (%c1) {
          memref.store %zero, %y[%i] : memref<1920xf32>
          scf.for %j = %c0 to %c16 step %c1 {
            %old = memref.load %y[%i] : memref<1920xf32>
            %a = memref.load %A[%i, %j] : memref<1920x16xf32>
            %next = arith.addf %old, %a : f32
            memref.store %next, %y[%i] : memref<1920xf32>
          }
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }
}

// CHECK-LABEL: func.func @owner_local_reduction_uses_single_channel_grain
// CHECK: codir.codelet
// CHECK-SAME: logical_worker_slice = [1]
// CHECK-SAME: partial_reduction
// CHECK-SAME: tile_shape = [1]
