/// Test: Nested memref with opaque func.call escape is a hard error.
/// MemrefNormalization must reject patterns it cannot fully raise to
/// N-dimensional memrefs, because silent skips produce broken output.
// RUN: not %carts-compile %s --O3 --arts-config %arts_config --pipeline raise-memref-dimensionality 2>&1 | %FileCheck %s

// CHECK: error: un-normalizable nested memref pattern

module {
  func.func private @touch_state(%state: memref<?xmemref<?xf32>>, %idx: index) -> f32 {
    %c0 = arith.constant 0 : index
    %cmp = arith.cmpi slt, %idx, %c0 : index
    cf.cond_br %cmp, ^bb1, ^bb2
  ^bb1:
    %zero = arith.constant 0.0 : f32
    return %zero : f32
  ^bb2:
    %row = memref.load %state[%c0] : memref<?xmemref<?xf32>>
    %value = memref.load %row[%idx] : memref<?xf32>
    return %value : f32
  }

  func.func @main() -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %out = memref.alloc() : memref<4xf32>
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%iv) : index = (%c0) to (%c4) step (%c1) {
          %wrapper = memref.alloca() : memref<memref<?xmemref<?xf32>>>
          %state = memref.alloc(%c4) : memref<?xmemref<?xf32>>
          memref.store %state, %wrapper[] : memref<memref<?xmemref<?xf32>>>
          scf.for %i = %c0 to %c4 step %c1 {
            %row_static = memref.alloc() : memref<4xf32>
            %row = memref.cast %row_static : memref<4xf32> to memref<?xf32>
            memref.store %row, %state[%i] : memref<?xmemref<?xf32>>
          }
          %state_view = memref.load %wrapper[] : memref<memref<?xmemref<?xf32>>>
          %result = func.call @touch_state(%state_view, %iv) : (memref<?xmemref<?xf32>>, index) -> f32
          memref.store %result, %out[%iv] : memref<4xf32>
          omp.yield
        }
      }
      omp.terminator
    }
    %r = memref.load %out[%c0] : memref<4xf32>
    memref.dealloc %out : memref<4xf32>
    return %r : f32
  }
}
