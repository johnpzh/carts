// RUN: %carts-compile %s --O3 --arts-config %arts_config --pipeline raise-memref-dimensionality | %FileCheck %s

// CHECK-LABEL: func.func @main
// CHECK-NOT: arts.undef : i1
// CHECK-NOT: memref<memref<?xmemref<?xmemref<?xf32>>>>
// CHECK: %[[CANON:.*]] = memref.alloc() : memref<2x3x4xf32>
// CHECK: %[[PTR0:.*]] = polygeist.memref2pointer %[[CANON]] : memref<2x3x4xf32> to !llvm.ptr
// CHECK: %[[ISNULL:.*]] = llvm.icmp "eq" %[[PTR0]], %{{.*}} : !llvm.ptr
// CHECK: scf.if %{{.*}} -> (i1) {
// CHECK: } else {
// CHECK: scf.yield %[[ISNULL]] : i1
// CHECK: }
// CHECK: memref.store {{.*}}, %[[CANON]][%{{.*}}, %{{.*}}, %{{.*}}] : memref<2x3x4xf32>
// CHECK: memref.load %[[CANON]][%c0, %c1, %c2] : memref<2x3x4xf32>

module {
  func.func @main() -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %null = llvm.mlir.zero : !llvm.ptr
    %true = arith.constant true

    %wrapper = memref.alloca() : memref<memref<?xmemref<?xmemref<?xf32>>>>
    %alias0 = memref.alloca() : memref<memref<?xmemref<?xmemref<?xf32>>>>
    %alias1 = memref.alloca() : memref<memref<?xmemref<?xmemref<?xf32>>>>
    %root = memref.alloc(%c2) : memref<?xmemref<?xmemref<?xf32>>>
    memref.store %root, %wrapper[] : memref<memref<?xmemref<?xmemref<?xf32>>>>
    %root_alias0 = memref.load %wrapper[] : memref<memref<?xmemref<?xmemref<?xf32>>>>
    memref.store %root_alias0, %alias0[] : memref<memref<?xmemref<?xmemref<?xf32>>>>
    %root_alias1 = memref.load %alias0[] : memref<memref<?xmemref<?xmemref<?xf32>>>>
    memref.store %root_alias1, %alias1[] : memref<memref<?xmemref<?xmemref<?xf32>>>>

    scf.for %i = %c0 to %c2 step %c1 {
      %mid = memref.alloc(%c3) : memref<?xmemref<?xf32>>
      memref.store %mid, %root[%i] : memref<?xmemref<?xmemref<?xf32>>>
      scf.for %j = %c0 to %c3 step %c1 {
        %leaf_static = memref.alloc() : memref<4xf32>
        %leaf = memref.cast %leaf_static : memref<4xf32> to memref<?xf32>
        %mid_view = memref.load %root[%i] : memref<?xmemref<?xmemref<?xf32>>>
        memref.store %leaf, %mid_view[%j] : memref<?xmemref<?xf32>>
      }
    }

    %ptr0 = polygeist.memref2pointer %root : memref<?xmemref<?xmemref<?xf32>>> to !llvm.ptr
    %is_null0 = llvm.icmp "eq" %ptr0, %null : !llvm.ptr
    %has_failure = scf.if %is_null0 -> (i1) {
      scf.yield %true : i1
    } else {
      %aliased_root = memref.load %alias1[] : memref<memref<?xmemref<?xmemref<?xf32>>>>
      %ptr1 = polygeist.memref2pointer %aliased_root : memref<?xmemref<?xmemref<?xf32>>> to !llvm.ptr
      %is_null1 = llvm.icmp "eq" %ptr1, %null : !llvm.ptr
      scf.yield %is_null1 : i1
    }
    %can_execute = arith.xori %has_failure, %true : i1

    scf.if %can_execute {
      scf.for %i = %c0 to %c2 step %c1 {
        scf.for %j = %c0 to %c3 step %c1 {
          %i_i32 = arith.index_cast %i : index to i32
          %j_i32 = arith.index_cast %j : index to i32
          scf.for %k = %c0 to %c4 step %c1 {
            %root_view = memref.load %wrapper[] : memref<memref<?xmemref<?xmemref<?xf32>>>>
            %mid_view = memref.load %root_view[%i] : memref<?xmemref<?xmemref<?xf32>>>
            %leaf_view = memref.load %mid_view[%j] : memref<?xmemref<?xf32>>
            %k_i32 = arith.index_cast %k : index to i32
            %ij = arith.addi %i_i32, %j_i32 : i32
            %ijk = arith.addi %ij, %k_i32 : i32
            %value = arith.sitofp %ijk : i32 to f32
            memref.store %value, %leaf_view[%k] : memref<?xf32>
          }
        }
      }
    }

    %result_root = memref.load %alias1[] : memref<memref<?xmemref<?xmemref<?xf32>>>>
    %result_mid = memref.load %result_root[%c0] : memref<?xmemref<?xmemref<?xf32>>>
    %result_leaf = memref.load %result_mid[%c1] : memref<?xmemref<?xf32>>
    %result = memref.load %result_leaf[%c2] : memref<?xf32>
    return %result : f32
  }
}
