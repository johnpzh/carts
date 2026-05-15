// RUN: %carts-compile %s --O3 --arts-config %arts_config --pipeline raise-memref-dimensionality | %FileCheck %s

// CHECK-LABEL: func.func @main
// CHECK-NOT: func.call @reduce_state
// CHECK-NOT: func.call @free_state
// CHECK-NOT: memref<memref<?xmemref<?xf64>>>
// CHECK-NOT: func.func private @reduce_state
// CHECK-NOT: func.func private @free_state
// CHECK: %[[STATE:.*]] = memref.alloc() : memref<4x4xf64>
// CHECK: memref.store {{.*}}, %[[STATE]][%{{.*}}, %{{.*}}] : memref<4x4xf64>
// CHECK: %[[SUM:.*]] = scf.for
// CHECK: memref.store %[[SUM]], %{{.*}}[%{{.*}}] : memref<4xf64>
// CHECK: memref.dealloc %[[STATE]] : memref<4x4xf64>

module {
  func.func private @reduce_state(%state: memref<?xmemref<?xf64>>, %n: index) -> f64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.000000e+00 : f64
    %sum = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %zero) -> (f64) {
      %row = memref.load %state[%i] : memref<?xmemref<?xf64>>
      %rowSum = scf.for %j = %c0 to %n step %c1 iter_args(%inner = %acc) -> (f64) {
        %value = memref.load %row[%j] : memref<?xf64>
        %next = arith.addf %inner, %value : f64
        scf.yield %next : f64
      }
      scf.yield %rowSum : f64
    }
    return %sum : f64
  }

  func.func private @free_state(%state: memref<?xmemref<?xf64>>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      %row = memref.load %state[%i] : memref<?xmemref<?xf64>>
      memref.dealloc %row : memref<?xf64>
    }
    memref.dealloc %state : memref<?xmemref<?xf64>>
    return
  }

  func.func @main() -> f64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %out = memref.alloc() : memref<4xf64>
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%iv) : index = (%c0) to (%c4) step (%c1) {
          %wrapper = memref.alloca() : memref<memref<?xmemref<?xf64>>>
          %state = memref.alloc(%c4) : memref<?xmemref<?xf64>>
          memref.store %state, %wrapper[] : memref<memref<?xmemref<?xf64>>>
          scf.for %i = %c0 to %c4 step %c1 {
            %rowStatic = memref.alloc() : memref<4xf64>
            %row = memref.cast %rowStatic : memref<4xf64> to memref<?xf64>
            memref.store %row, %state[%i] : memref<?xmemref<?xf64>>
          }
          scf.for %i = %c0 to %c4 step %c1 {
            %i32 = arith.index_cast %i : index to i32
            scf.for %j = %c0 to %c4 step %c1 {
              %j32 = arith.index_cast %j : index to i32
              %stateView = memref.load %wrapper[] : memref<memref<?xmemref<?xf64>>>
              %row = memref.load %stateView[%i] : memref<?xmemref<?xf64>>
              %mul0 = arith.muli %i32, %c2_i32 : i32
              %sum0 = arith.addi %mul0, %j32 : i32
              %sum1 = arith.addi %sum0, %c3_i32 : i32
              %value = arith.sitofp %sum1 : i32 to f64
              memref.store %value, %row[%j] : memref<?xf64>
            }
          }
          %stateForReduce = memref.load %wrapper[] : memref<memref<?xmemref<?xf64>>>
          %sum = func.call @reduce_state(%stateForReduce, %c4) : (memref<?xmemref<?xf64>>, index) -> f64
          memref.store %sum, %out[%iv] : memref<4xf64>
          %stateForFree = memref.load %wrapper[] : memref<memref<?xmemref<?xf64>>>
          func.call @free_state(%stateForFree, %c4) : (memref<?xmemref<?xf64>>, index) -> ()
          omp.yield
        }
      }
      omp.terminator
    }
    %result = memref.load %out[%c0] : memref<4xf64>
    memref.dealloc %out : memref<4xf64>
    return %result : f64
  }
}
