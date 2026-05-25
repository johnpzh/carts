// RUN: %carts-compile %s --pass-pipeline='builtin.module(convert-sde-to-codir,verify-codir,convert-codir-to-arts)' \
// RUN:   | %FileCheck %s --implicit-check-not=codir.codelet --implicit-check-not=sde.

module {
  func.func @tokenizes_aligned_write_read_barrier(%A: memref<8x8xf32>,
                                                  %B: memref<8x8xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %tmp = sde.mu_alloc : memref<8x8xf32>

    sde.cu_region <parallel> {
      sde.su_distribute <blocked> {
        sde.su_iterate (%c0) to (%c8) step (%c1) classification(<elementwise>) {
        ^bb0(%i: index):
          scf.for %j = %c0 to %c8 step %c1 {
            %v = memref.load %A[%i, %j] : memref<8x8xf32>
            memref.store %v, %tmp[%i, %j] : memref<8x8xf32>
          }
          sde.yield
        } {iterationTopology = #sde.iteration_topology<owner_strip>,
           logicalWorkerSlice = [4, 8], pattern = #sde.pattern<uniform>,
           physicalBlockShape = [4, 8], physicalOwnerDims = [0]}
      }

      sde.su_barrier {barrierEliminated, barrierReason = #sde.barrier_reason<required_memory>}

      sde.su_distribute <blocked> {
        sde.su_iterate (%c0) to (%c8) step (%c1) classification(<elementwise>) {
        ^bb0(%i: index):
          scf.for %j = %c0 to %c8 step %c1 {
            %v = memref.load %tmp[%i, %j] : memref<8x8xf32>
            memref.store %v, %B[%i, %j] : memref<8x8xf32>
          }
          sde.yield
        } {iterationTopology = #sde.iteration_topology<owner_strip>,
           logicalWorkerSlice = [4, 8], pattern = #sde.pattern<uniform>,
           physicalBlockShape = [4, 8], physicalOwnerDims = [0]}
      }
      sde.yield
    }
    return
  }

  func.func @preserves_barrier_for_mismatched_access_window(%A: memref<8x8xf32>,
                                                            %B: memref<8x8xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %tmp = sde.mu_alloc : memref<8x8xf32>

    sde.cu_region <parallel> {
      sde.su_iterate (%c0) to (%c8) step (%c1) classification(<elementwise>) {
      ^bb0(%i: index):
        scf.for %j = %c0 to %c8 step %c1 {
          %v = memref.load %A[%i, %j] : memref<8x8xf32>
          memref.store %v, %tmp[%i, %j] : memref<8x8xf32>
        }
        sde.yield
      } {iterationTopology = #sde.iteration_topology<owner_strip>,
         logicalWorkerSlice = [4, 8], pattern = #sde.pattern<uniform>,
         physicalBlockShape = [4, 8], physicalOwnerDims = [0]}

      // If an eliminated required-memory barrier cannot be reconstructed as a
      // token-local plan, convert-sde-to-codir must fail closed by preserving
      // the barrier for ARTS materialization.
      sde.su_barrier {barrierEliminated, barrierReason = #sde.barrier_reason<required_memory>}

      sde.su_iterate (%c0) to (%c8) step (%c1) classification(<elementwise>) {
      ^bb0(%i: index):
        scf.for %j = %c0 to %c8 step %c1 {
          %v = memref.load %tmp[%j, %i] : memref<8x8xf32>
          memref.store %v, %B[%i, %j] : memref<8x8xf32>
        }
        sde.yield
      } {iterationTopology = #sde.iteration_topology<owner_strip>,
         logicalWorkerSlice = [4, 8], pattern = #sde.pattern<uniform>,
         physicalBlockShape = [4, 8], physicalOwnerDims = [0]}
      sde.yield
    }
    return
  }
}

// CHECK-LABEL: func.func @tokenizes_aligned_write_read_barrier
// CHECK: scf.for
// CHECK: arts.db_acquire[<out>]
// CHECK: arts.edt <task>
// CHECK: }
// CHECK-NOT: arts.barrier
// CHECK: scf.for
// CHECK: arts.db_acquire[<in>]
// CHECK: arts.edt <task>
// CHECK: }
// CHECK: arts.barrier

// CHECK-LABEL: func.func @preserves_barrier_for_mismatched_access_window
// CHECK: scf.for
// CHECK: arts.edt <task>
// CHECK: }
// CHECK: arts.barrier
// CHECK: scf.for
