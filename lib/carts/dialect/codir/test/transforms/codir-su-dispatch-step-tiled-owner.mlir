// RUN: %carts-compile %s --pass-pipeline='builtin.module(convert-sde-to-codir,verify-codir)' \
// RUN:   | %FileCheck %s --implicit-check-not=sde.

// SDE tiling may already rewrite the scheduling-unit step to the physical
// owner block. CODIR must not multiply that step by the block shape again, or
// the launch loop collapses row-strip EDTs into one oversized codelet.

module {
  func.func @su_dispatch_step_keeps_sde_tiled_owner_step(%A: memref<4800x4800xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c75 = arith.constant 75 : index
    %c4800 = arith.constant 4800 : index
    sde.cu_region <parallel> {
      sde.su_iterate (%c0) to (%c4800) step (%c75) classification(<matmul>) {
      ^bb0(%i: index):
        %v = memref.load %A[%i, %c0] : memref<4800x4800xf32>
        memref.store %v, %A[%i, %c0] : memref<4800x4800xf32>
        sde.yield
      } {iterationTopology = #sde.iteration_topology<owner_strip>,
         logicalWorkerSlice = [75, 4800], pattern = #sde.pattern<matmul>,
         physicalBlockShape = [75, 4800], physicalOwnerDims = [0]}
      sde.yield
    }
    return
  }

  func.func @su_dispatch_step_uses_logical_worker_slice(%A: memref<16320xf64>, %B: memref<16320xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16320 = arith.constant 16320 : index
    sde.cu_region <parallel> {
      sde.su_iterate (%c0) to (%c16320) step (%c1) classification(<reduction>) {
      ^bb0(%i: index):
        %v = memref.load %B[%i] : memref<16320xf64>
        memref.store %v, %A[%i] : memref<16320xf64>
        sde.yield
      } {iterationTopology = #sde.iteration_topology<owner_strip>,
         logicalWorkerSlice = [255], pattern = #sde.pattern<reduction>}
      sde.yield
    }
    return
  }

  func.func @su_owner_tile_uses_multidim_logical_slice(%A: memref<19200x19200xf32>, %B: memref<19200x19200xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c19199 = arith.constant 19199 : index
    sde.cu_region <parallel> {
      sde.su_iterate (%c1, %c1) to (%c19199, %c19199) step (%c1, %c1) classification(<stencil>) {
      ^bb0(%i: index, %j: index):
        %v = memref.load %B[%i, %j] : memref<19200x19200xf32>
        memref.store %v, %A[%i, %j] : memref<19200x19200xf32>
        sde.yield
      } {iterationTopology = #sde.iteration_topology<owner_tile>,
         logicalWorkerSlice = [2400, 2400], pattern = #sde.pattern<stencil_tiling_nd>}
      sde.yield
    }
    return
  }

  func.func @su_dispatch_step_uses_owner_strip_multidim_logical_slice(%A: memref<2304000xf32>, %B: memref<2304000x30xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2304000 = arith.constant 2304000 : index
    sde.cu_region <parallel> {
      sde.su_iterate (%c0) to (%c2304000) step (%c1) classification(<reduction>) {
      ^bb0(%i: index):
        %v = memref.load %B[%i, %c0] : memref<2304000x30xf32>
        memref.store %v, %A[%i] : memref<2304000xf32>
        sde.yield
      } {iterationTopology = #sde.iteration_topology<owner_strip>,
         logicalWorkerSlice = [36000, 30], pattern = #sde.pattern<reduction>}
      sde.yield
    }
    return
  }

  func.func @su_dispatch_step_uses_trailing_owner_dim_logical_slice(%A: memref<16x16x64xf64>, %B: memref<16x16x64xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c14 = arith.constant 14 : index
    %c62 = arith.constant 62 : index
    sde.cu_region <parallel> {
      sde.su_iterate (%c2) to (%c62) step (%c1) classification(<stencil>) {
      ^bb0(%k: index):
        scf.for %j = %c2 to %c14 step %c1 {
          scf.for %i = %c2 to %c14 step %c1 {
            %v = memref.load %A[%i, %j, %k] : memref<16x16x64xf64>
            memref.store %v, %B[%i, %j, %k] : memref<16x16x64xf64>
          }
        }
        sde.yield
      } {iterationTopology = #sde.iteration_topology<owner_strip>,
         logicalWorkerSlice = [16, 16, 4], ownerDims = [0, 1, 2],
         pattern = #sde.pattern<cross_dim_stencil_3d>}
      sde.yield
    }
    return
  }

  func.func @su_localizes_private_alloca(%A: memref<8xf32>, %B: memref<8xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %scratch = memref.alloca() : memref<4xf32>
    sde.cu_region <parallel> {
      sde.su_iterate (%c0) to (%c8) step (%c1) classification(<elementwise>) {
      ^bb0(%i: index):
        %v = memref.load %A[%i] : memref<8xf32>
        memref.store %v, %scratch[%c0] : memref<4xf32>
        %w = memref.load %scratch[%c0] : memref<4xf32>
        memref.store %w, %B[%i] : memref<8xf32>
        sde.yield
      } {iterationTopology = #sde.iteration_topology<owner_strip>,
         logicalWorkerSlice = [4], pattern = #sde.pattern<uniform>}
      sde.yield
    }
    return
  }
}

// CHECK-LABEL: func.func @su_dispatch_step_keeps_sde_tiled_owner_step
// CHECK-DAG: %[[C75:.*]] = arith.constant 75 : index
// CHECK-NOT: arith.muli %[[C75]], %[[C75]]
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[C75]]
// CHECK: codir.codelet

// CHECK-LABEL: func.func @su_dispatch_step_uses_logical_worker_slice
// CHECK-DAG: %[[C255:.*]] = arith.constant 255 : index
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[C255]]
// CHECK: codir.codelet

// CHECK-LABEL: func.func @su_owner_tile_uses_multidim_logical_slice
// CHECK-DAG: %[[C2400_0:.*]] = arith.constant 2400 : index
// CHECK-DAG: %[[C2400_1:.*]] = arith.constant 2400 : index
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[C2400_0]]
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[C2400_1]]
// CHECK: codir.codelet
// CHECK: arith.minui
// CHECK: scf.for
// CHECK: arith.minui
// CHECK: scf.for

// CHECK-LABEL: func.func @su_dispatch_step_uses_owner_strip_multidim_logical_slice
// CHECK-DAG: %[[C36000:.*]] = arith.constant 36000 : index
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[C36000]]
// CHECK: codir.codelet

// CHECK-LABEL: func.func @su_dispatch_step_uses_trailing_owner_dim_logical_slice
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[C4]]
// CHECK: codir.codelet

// CHECK-LABEL: func.func @su_localizes_private_alloca
// CHECK: codir.codelet deps(%{{.*}}, %{{.*}} : memref<8xf32>, memref<8xf32>) params
// CHECK: memref.alloca() : memref<4xf32>
