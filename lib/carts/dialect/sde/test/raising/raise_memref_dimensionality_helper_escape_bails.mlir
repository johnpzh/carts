// RUN: %carts-compile %s --O3 --arts-config %arts_config --pipeline raise-memref-dimensionality | %FileCheck %s

// CHECK-LABEL: func.func @main
// CHECK-NOT: func.call @init_state
// CHECK-NOT: func.func private @init_state
// CHECK-NOT: memref<memref<?xmemref<?xmemref<?xf32>>>>
// CHECK: %[[STATE:.*]] = memref.alloc() : memref<2x3x4xf32>
// CHECK: memref.store {{.*}}, %[[STATE]][%{{.*}}, %{{.*}}, %{{.*}}] : memref<2x3x4xf32>
// CHECK: %[[RESULT:.*]] = memref.load %[[STATE]][%c0, %c1, %c2] : memref<2x3x4xf32>
// CHECK: return %[[RESULT]] : f32

module {
  func.func private @init_state(%state: memref<?xmemref<?xmemref<?xf32>>>,
                                %m: index, %n: index, %k: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.000000e+00 : f32
    scf.for %i = %c0 to %m step %c1 {
      %plane = memref.load %state[%i] : memref<?xmemref<?xmemref<?xf32>>>
      scf.for %j = %c0 to %n step %c1 {
        %row = memref.load %plane[%j] : memref<?xmemref<?xf32>>
        scf.for %l = %c0 to %k step %c1 {
          memref.store %zero, %row[%l] : memref<?xf32>
        }
      }
    }
    return
  }

  func.func @main() -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index

    %wrapper = memref.alloca() : memref<memref<?xmemref<?xmemref<?xf32>>>>
    %root = memref.alloc(%c2) : memref<?xmemref<?xmemref<?xf32>>>
    memref.store %root, %wrapper[] : memref<memref<?xmemref<?xmemref<?xf32>>>>

    scf.for %i = %c0 to %c2 step %c1 {
      %plane = memref.alloc(%c3) : memref<?xmemref<?xf32>>
      memref.store %plane, %root[%i] : memref<?xmemref<?xmemref<?xf32>>>
      scf.for %j = %c0 to %c3 step %c1 {
        %plane_view = memref.load %root[%i] : memref<?xmemref<?xmemref<?xf32>>>
        %row_static = memref.alloc() : memref<4xf32>
        %row = memref.cast %row_static : memref<4xf32> to memref<?xf32>
        memref.store %row, %plane_view[%j] : memref<?xmemref<?xf32>>
      }
    }

    %state = memref.load %wrapper[] : memref<memref<?xmemref<?xmemref<?xf32>>>>
    func.call @init_state(%state, %c2, %c3, %c4) : (memref<?xmemref<?xmemref<?xf32>>>, index, index, index) -> ()

    %result_state = memref.load %wrapper[] : memref<memref<?xmemref<?xmemref<?xf32>>>>
    %result_plane = memref.load %result_state[%c0] : memref<?xmemref<?xmemref<?xf32>>>
    %result_row = memref.load %result_plane[%c1] : memref<?xmemref<?xf32>>
    %result = memref.load %result_row[%c2] : memref<?xf32>
    return %result : f32
  }
}
