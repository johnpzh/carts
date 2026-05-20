// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from sde-planning --pipeline sde-to-codir --mlir-print-ir-after-all 2>&1 | %FileCheck %s
// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_multinode_64x64.cfg --start-from sde-planning --pipeline sde-to-codir --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=MN64

// Row-local reductions inside one owner iteration are not cross-iteration
// reductions. SDE should keep the owner dimension chunkable so CODIR/ARTS do
// not materialize one EDT per row.

// CHECK-LABEL: // -----// IR Dump After PatternAnalysis (sde-pattern-analysis) //----- //
// CHECK: func.func @row_local_pipeline
// CHECK: sde.su_iterate (%c0) to (%c1024) step (%c1) classification(<elementwise_pipeline>) {
// CHECK: func.func @shifted_self_read_not_owner_local
// CHECK: sde.su_iterate (%c0) to (%c1023) step (%c1) classification(<reduction>) {
// CHECK-LABEL: // -----// IR Dump After Tiling (tiling) //----- //
// CHECK: func.func @row_local_pipeline
// CHECK: %[[STEP:.*]] = arith.muli %c1, %c128{{(_[0-9]+)?}} : index
// CHECK: sde.su_iterate (%c0) to (%c1024) step (%[[STEP]]) classification(<elementwise_pipeline>) {
// CHECK: func.func @shifted_self_read_not_owner_local
// CHECK: sde.su_iterate (%c0) to (%c1023) step (%c1) classification(<reduction>) {
// CHECK-LABEL: // -----// IR Dump After ConvertSdeToCodir (convert-sde-to-codir) //----- //
// CHECK: func.func @row_local_pipeline
// CHECK: codir.codelet
// CHECK-SAME: logical_worker_slice = [128, 256]
// CHECK-SAME: pattern = #codir.pattern<elementwise_pipeline>
// CHECK-SAME: tile_shape = [128, 256]

// MN64-LABEL: // -----// IR Dump After Tiling (tiling) //----- //
// MN64: func.func @row_local_pipeline
// MN64: sde.su_iterate (%c0) to (%c1024) step (%c1) reduction_strategy(<local_accumulate>) classification(<elementwise_pipeline>) {
// MN64: physicalBlockShape = [1, 256]
// MN64-LABEL: // -----// IR Dump After ConvertSdeToCodir (convert-sde-to-codir) //----- //
// MN64: func.func @row_local_pipeline
// MN64: codir.codelet
// MN64-SAME: logical_worker_slice = [1, 256]
// MN64-SAME: pattern = #codir.pattern<elementwise_pipeline>
// MN64-SAME: tile_shape = [1, 256]

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @row_local_pipeline(%x: memref<1024x256xf32>, %gamma: memref<256xf32>, %beta: memref<256xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c1024 = arith.constant 1024 : index
    %zero = arith.constant 0.0 : f32
    %one = arith.constant 1.0 : f32
    %eps = arith.constant 9.99999974E-6 : f32
    %hidden = arith.constant 2.560000e+02 : f32
    scf.for %rep = %c0 to %c128 step %c1 {
      sde.cu_region <parallel> {
        sde.su_iterate (%c0) to (%c1024) step (%c1) {
        ^bb0(%b: index):
          %mean = memref.alloca() : memref<f32>
          %var = memref.alloca() : memref<f32>
          memref.store %zero, %mean[] : memref<f32>
          memref.store %zero, %var[] : memref<f32>
          scf.for %h = %c0 to %c256 step %c1 {
            %v = memref.load %x[%b, %h] : memref<1024x256xf32>
            %acc = memref.load %mean[] : memref<f32>
            %next = arith.addf %acc, %v : f32
            memref.store %next, %mean[] : memref<f32>
          }
          %sum = memref.load %mean[] : memref<f32>
          %avg = arith.divf %sum, %hidden : f32
          memref.store %zero, %var[] : memref<f32>
          scf.for %h = %c0 to %c256 step %c1 {
            %v = memref.load %x[%b, %h] : memref<1024x256xf32>
            %diff = arith.subf %v, %avg : f32
            %sq = arith.mulf %diff, %diff : f32
            %acc = memref.load %var[] : memref<f32>
            %next = arith.addf %acc, %sq : f32
            memref.store %next, %var[] : memref<f32>
          }
          %var_sum = memref.load %var[] : memref<f32>
          %var_avg = arith.divf %var_sum, %hidden : f32
          %var_eps = arith.addf %var_avg, %eps : f32
          %std = math.sqrt %var_eps : f32
          %inv_std = arith.divf %one, %std : f32
          scf.for %h = %c0 to %c256 step %c1 {
            %v = memref.load %x[%b, %h] : memref<1024x256xf32>
            %centered = arith.subf %v, %avg : f32
            %norm = arith.mulf %centered, %inv_std : f32
            %g = memref.load %gamma[%h] : memref<256xf32>
            %scaled = arith.mulf %norm, %g : f32
            %offset = memref.load %beta[%h] : memref<256xf32>
            %out = arith.addf %scaled, %offset : f32
            memref.store %out, %x[%b, %h] : memref<1024x256xf32>
          }
          sde.yield
        }
        sde.yield
      }
    }
    return
  }

  func.func @shifted_self_read_not_owner_local(%x: memref<1024xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c1023 = arith.constant 1023 : index
    %zero = arith.constant 0.0 : f32
    scf.for %rep = %c0 to %c128 step %c1 {
      sde.cu_region <parallel> {
        sde.su_iterate (%c0) to (%c1023) step (%c1) {
        ^bb0(%b: index):
          %bp1 = arith.addi %b, %c1 : index
          memref.store %zero, %x[%b] : memref<1024xf32>
          scf.for %h = %c0 to %c256 step %c1 {
            %neighbor = memref.load %x[%bp1] : memref<1024xf32>
            %acc = memref.load %x[%b] : memref<1024xf32>
            %next = arith.addf %acc, %neighbor : f32
            memref.store %next, %x[%b] : memref<1024xf32>
          }
          sde.yield
        }
        sde.yield
      }
    }
    return
  }
}
