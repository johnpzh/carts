// RUN: not %carts-compile %s --O3 --arts-config %arts_config --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s

// Verify that LoopInterchange does NOT reorder stencil loops. The pass
// only targets matmul patterns (j-k to k-j for stride-1 B access). Stencil
// loops with their original iteration order must pass through unchanged.

// CHECK-LABEL: // -----// IR Dump After LoopInterchange (loop-interchange) //----- //
// CHECK: sde.su_iterate
// CHECK-SAME: classification(<stencil>)
// The inner scf.for loop must remain in the original order:
// CHECK: scf.for %[[J:[A-Za-z0-9_]+]] =
// CHECK: arith.subi %{{.+}}, %c1
// CHECK: arith.addi %{{.+}}, %c1
// CHECK: arith.subi %[[J]], %c1
// CHECK: arith.addi %[[J]], %c1
// CHECK: memref.load
// CHECK: memref.store
// Result-bearing stencil loops must also remain untouched. The pass cannot
// erase and rebuild loops whose results are used by surrounding tensor state.
// CHECK: func.func @result_bearing_stencil
// CHECK: sde.su_iterate
// CHECK: memref.store
// CHECK: scf.for {{.*}} iter_args
// CHECK: scf.for {{.*}} iter_args
// Result-bearing loops that otherwise look like promoted matmul accumulators
// must also remain untouched because the rewrite cannot replace the outer loop
// result.
// CHECK: func.func @result_bearing_promoted_accumulator
// CHECK: sde.su_iterate
// CHECK: scf.for {{.*}} iter_args
// CHECK: scf.for {{.*}} iter_args
// CHECK: error: 'sde.su_iterate' op cannot directly materialize result-producing su_iterate reductions

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main(%A: memref<64x64xf64>, %B: memref<64x64xf64>) {
    %c1 = arith.constant 1 : index
    %c63 = arith.constant 63 : index
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c1) to (%c63) step (%c1) {
          scf.for %j = %c1 to %c63 step %c1 {
            %im1 = arith.subi %i, %c1 : index
            %ip1 = arith.addi %i, %c1 : index
            %jm1 = arith.subi %j, %c1 : index
            %jp1 = arith.addi %j, %c1 : index
            %n = memref.load %A[%im1, %j] : memref<64x64xf64>
            %s = memref.load %A[%ip1, %j] : memref<64x64xf64>
            %w = memref.load %A[%i, %jm1] : memref<64x64xf64>
            %e = memref.load %A[%i, %jp1] : memref<64x64xf64>
            %c = memref.load %A[%i, %j] : memref<64x64xf64>
            %s0 = arith.addf %n, %s : f64
            %s1 = arith.addf %w, %e : f64
            %s2 = arith.addf %s0, %s1 : f64
            %sum = arith.addf %s2, %c : f64
            memref.store %sum, %B[%i, %j] : memref<64x64xf64>
          }
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }

  func.func @result_bearing_stencil(%A: memref<8x8x8xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c7 = arith.constant 7 : index
    %scratch = memref.alloca() : memref<f64>
    %empty = tensor.empty() : tensor<2xf64>
    %zero = arith.constant 0.0 : f64
    %init = tensor.insert %zero into %empty[%c0] : tensor<2xf64>
    %init_1 = tensor.insert %zero into %init[%c1] : tensor<2xf64>
    %result = sde.cu_region <parallel> iter_args(%arg0 = %init_1 : tensor<2xf64>) -> (tensor<2xf64>) {
      %iter = sde.su_iterate (%c1) to (%c7) step (%c1) classification(<stencil>) iter_args(%arg2 = %arg0 : tensor<2xf64>) -> (tensor<2xf64>) {
      ^bb0(%i: index, %arg3: tensor<2xf64>):
        memref.store %zero, %scratch[] : memref<f64>
        %j_result = scf.for %j = %c1 to %c7 step %c1 iter_args(%j_arg = %arg3) -> (tensor<2xf64>) {
          %k_result = scf.for %k = %c1 to %c7 step %c1 iter_args(%k_arg = %j_arg) -> (tensor<2xf64>) {
            %loaded = memref.load %A[%i, %j, %k] : memref<8x8x8xf64>
            memref.store %loaded, %scratch[] : memref<f64>
            %next = tensor.insert %loaded into %k_arg[%c1] : tensor<2xf64>
            scf.yield %next : tensor<2xf64>
          }
          scf.yield %k_result : tensor<2xf64>
        }
        sde.yield %j_result : tensor<2xf64>
      } {accessMaxOffsets = [1, 2, 1], accessMinOffsets = [-1, -2, -1], ownerDims = [0, 1, 2], spatialDims = [0, 1, 2], writeFootprint = [1, 1, 1]}
      sde.yield %iter : tensor<2xf64>
    }
    %out = tensor.extract %result[%c0] : tensor<2xf64>
    memref.store %out, %A[%c0, %c0, %c0] : memref<8x8x8xf64>
    return
  }

  func.func @result_bearing_promoted_accumulator(%A: memref<8x8xf64>, %B: memref<8x8xf64>, %C: memref<8x8xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant 0.0 : f64
    %empty = tensor.empty() : tensor<1xf64>
    %init = tensor.insert %zero into %empty[%c0] : tensor<1xf64>
    %result = sde.cu_region <parallel> iter_args(%arg0 = %init : tensor<1xf64>) -> (tensor<1xf64>) {
      %iter = sde.su_iterate (%c0) to (%c8) step (%c1) classification(<matmul>) iter_args(%arg2 = %arg0 : tensor<1xf64>) -> (tensor<1xf64>) {
      ^bb0(%i: index, %arg3: tensor<1xf64>):
        %j_result = scf.for %j = %c0 to %c8 step %c1 iter_args(%j_arg = %arg3) -> (tensor<1xf64>) {
          %sum = scf.for %k = %c0 to %c8 step %c1 iter_args(%acc = %zero) -> (f64) {
            %a = memref.load %A[%i, %k] : memref<8x8xf64>
            %b = memref.load %B[%k, %j] : memref<8x8xf64>
            %prod = arith.mulf %a, %b : f64
            %next = arith.addf %acc, %prod : f64
            scf.yield %next : f64
          }
          %next_state = tensor.insert %sum into %j_arg[%c0] : tensor<1xf64>
          memref.store %sum, %C[%i, %j] : memref<8x8xf64>
          scf.yield %next_state : tensor<1xf64>
        }
        sde.yield %j_result : tensor<1xf64>
      }
      sde.yield %iter : tensor<1xf64>
    }
    %out = tensor.extract %result[%c0] : tensor<1xf64>
    memref.store %out, %C[%c0, %c0] : memref<8x8xf64>
    return
  }
}
