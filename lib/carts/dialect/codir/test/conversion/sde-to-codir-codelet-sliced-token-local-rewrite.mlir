// RUN: %carts-compile %s --pass-pipeline='builtin.module(convert-sde-to-codir,verify-codir)' | %FileCheck %s --implicit-check-not=sde.cu_codelet

module {
  func.func @sde_sliced_token_global_index_to_local(%mem: memref<8xi32>,
                                                    %global_i: index,
                                                    %scale: i32) {
    %c4 = arith.constant 4 : index
    %token = sde.mu_token <readwrite> %mem[%c4] size[%c4]
      : memref<8xi32> -> !sde.token<memref<4xi32, strided<[1], offset: 4>>>

    sde.cu_codelet (%token : !sde.token<memref<4xi32, strided<[1], offset: 4>>>)
      captures(%global_i, %scale : index, i32) {
    ^bb0(%view: memref<4xi32, strided<[1], offset: 4>>, %global_i_arg: index, %scale_arg: i32):
      %value = memref.load %view[%global_i_arg] : memref<4xi32, strided<[1], offset: 4>>
      %updated = arith.addi %value, %scale_arg : i32
      memref.store %updated, %view[%global_i_arg] : memref<4xi32, strided<[1], offset: 4>>
      sde.yield
    }
    return
  }

  func.func @sde_sliced_token_dynamic_offset_param(%mem: memref<16xi32>,
                                                   %off: index,
                                                   %global_i: index) {
    %c4 = arith.constant 4 : index
    %token = sde.mu_token <read> %mem[%off] size[%c4]
      : memref<16xi32> -> !sde.token<memref<4xi32, strided<[1], offset: ?>>>

    sde.cu_codelet (%token : !sde.token<memref<4xi32, strided<[1], offset: ?>>>)
      captures(%global_i : index) {
    ^bb0(%view: memref<4xi32, strided<[1], offset: ?>>, %global_i_arg: index):
      %c0 = arith.constant 0 : index
      %value = memref.load %view[%global_i_arg] : memref<4xi32, strided<[1], offset: ?>>
      %tmp = memref.alloca() : memref<1xi32>
      memref.store %value, %tmp[%c0] : memref<1xi32>
      sde.yield
    }
    return
  }

  func.func @sde_sliced_token_2d_global_indices_to_local(%mem: memref<8x8xi32>,
                                                         %global_i: index,
                                                         %global_j: index) {
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %token = sde.mu_token <readwrite> %mem[%c2, %c4] size[%c4, %c4]
      : memref<8x8xi32> -> !sde.token<memref<4x4xi32, strided<[8, 1], offset: 20>>>

    sde.cu_codelet (%token : !sde.token<memref<4x4xi32, strided<[8, 1], offset: 20>>>)
      captures(%global_i, %global_j : index, index) {
    ^bb0(%view: memref<4x4xi32, strided<[8, 1], offset: 20>>, %global_i_arg: index, %global_j_arg: index):
      %one = arith.constant 1 : i32
      %value = memref.load %view[%global_i_arg, %global_j_arg] : memref<4x4xi32, strided<[8, 1], offset: 20>>
      %updated = arith.addi %value, %one : i32
      memref.store %updated, %view[%global_i_arg, %global_j_arg] : memref<4x4xi32, strided<[8, 1], offset: 20>>
      sde.yield
    }
    return
  }

  func.func @sde_sliced_token_2d_dynamic_offsets(%mem: memref<16x16xi32>,
                                                 %row_off: index,
                                                 %col_off: index,
                                                 %global_i: index,
                                                 %global_j: index) {
    %c4 = arith.constant 4 : index
    %token = sde.mu_token <read> %mem[%row_off, %col_off] size[%c4, %c4]
      : memref<16x16xi32> -> !sde.token<memref<4x4xi32, strided<[16, 1], offset: ?>>>

    sde.cu_codelet (%token : !sde.token<memref<4x4xi32, strided<[16, 1], offset: ?>>>)
      captures(%global_i, %global_j : index, index) {
    ^bb0(%view: memref<4x4xi32, strided<[16, 1], offset: ?>>, %global_i_arg: index, %global_j_arg: index):
      %c0 = arith.constant 0 : index
      %value = memref.load %view[%global_i_arg, %global_j_arg] : memref<4x4xi32, strided<[16, 1], offset: ?>>
      %tmp = memref.alloca() : memref<1xi32>
      memref.store %value, %tmp[%c0] : memref<1xi32>
      sde.yield
    }
    return
  }
}

// CHECK-LABEL: func.func @sde_sliced_token_global_index_to_local
// CHECK: codir.codelet deps(%[[DEP:.*]] : memref<4xi32, strided<[1], offset: 4>>) params(%[[GLOBAL:.*]], %[[SCALE:.*]] : index, i32)
// CHECK: ^bb0(%[[VIEW:.*]]: memref<4xi32, strided<[1], offset: 4>>, %[[I:.*]]: index, %[[SCALE_ARG:.*]]: i32):
// CHECK: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[LOCAL_LOAD:.*]] = arith.subi %[[I]], %[[C4]] : index
// CHECK: %[[VALUE:.*]] = memref.load %[[VIEW]][%[[LOCAL_LOAD]]] : memref<4xi32, strided<[1], offset: 4>>
// CHECK: %[[UPDATED:.*]] = arith.addi %[[VALUE]], %[[SCALE_ARG]] : i32
// CHECK: %[[LOCAL_STORE:.*]] = arith.subi %[[I]], %[[C4]] : index
// CHECK: memref.store %[[UPDATED]], %[[VIEW]][%[[LOCAL_STORE]]] : memref<4xi32, strided<[1], offset: 4>>
// CHECK: codir.yield

// CHECK-LABEL: func.func @sde_sliced_token_dynamic_offset_param
// CHECK: codir.codelet deps(%[[DEP_DYN:.*]] : memref<4xi32, strided<[1], offset: ?>>) params(%[[GLOBAL_DYN:.*]], %[[OFF_DYN:.*]] : index, index)
// CHECK: ^bb0(%[[VIEW_DYN:.*]]: memref<4xi32, strided<[1], offset: ?>>, %[[I_DYN:.*]]: index, %[[OFF_ARG:.*]]: index):
// CHECK: %[[C0_DYN:.*]] = arith.constant 0 : index
// CHECK: %[[LOCAL_DYN:.*]] = arith.subi %[[I_DYN]], %[[OFF_ARG]] : index
// CHECK: %[[VALUE_DYN:.*]] = memref.load %[[VIEW_DYN]][%[[LOCAL_DYN]]] : memref<4xi32, strided<[1], offset: ?>>
// CHECK: %[[TMP_DYN:.*]] = memref.alloca() : memref<1xi32>
// CHECK: memref.store %[[VALUE_DYN]], %[[TMP_DYN]][%[[C0_DYN]]] : memref<1xi32>
// CHECK: codir.yield

// CHECK-LABEL: func.func @sde_sliced_token_2d_global_indices_to_local
// CHECK: codir.codelet deps(%[[DEP_2D:.*]] : memref<4x4xi32
// CHECK-SAME: params(%[[GLOBAL_I_2D:.*]], %[[GLOBAL_J_2D:.*]] : index, index)
// CHECK: ^bb0(%[[VIEW_2D:.*]]: memref<4x4xi32{{.*}}, %[[I_2D:.*]]: index, %[[J_2D:.*]]: index):
// CHECK: %[[C4_2D:.*]] = arith.constant 4 : index
// CHECK: %[[C2_2D:.*]] = arith.constant 2 : index
// CHECK: %[[ONE_2D:.*]] = arith.constant 1 : i32
// CHECK: %[[LOCAL_I_LOAD:.*]] = arith.subi %[[I_2D]], %[[C2_2D]] : index
// CHECK: %[[LOCAL_J_LOAD:.*]] = arith.subi %[[J_2D]], %[[C4_2D]] : index
// CHECK: %[[VALUE_2D:.*]] = memref.load %[[VIEW_2D]][%[[LOCAL_I_LOAD]], %[[LOCAL_J_LOAD]]]
// CHECK: %[[UPDATED_2D:.*]] = arith.addi %[[VALUE_2D]], %[[ONE_2D]] : i32
// CHECK: %[[LOCAL_I_STORE:.*]] = arith.subi %[[I_2D]], %[[C2_2D]] : index
// CHECK: %[[LOCAL_J_STORE:.*]] = arith.subi %[[J_2D]], %[[C4_2D]] : index
// CHECK: memref.store %[[UPDATED_2D]], %[[VIEW_2D]][%[[LOCAL_I_STORE]], %[[LOCAL_J_STORE]]]
// CHECK: codir.yield

// CHECK-LABEL: func.func @sde_sliced_token_2d_dynamic_offsets
// CHECK: codir.codelet deps(%[[DEP_DYN_2D:.*]] : memref<4x4xi32
// CHECK-SAME: params(%[[GLOBAL_I_DYN_2D:.*]], %[[GLOBAL_J_DYN_2D:.*]], %[[ROW_OFF_DYN_2D:.*]], %[[COL_OFF_DYN_2D:.*]] : index, index, index, index)
// CHECK: ^bb0(%[[VIEW_DYN_2D:.*]]: memref<4x4xi32{{.*}}, %[[I_DYN_2D:.*]]: index, %[[J_DYN_2D:.*]]: index, %[[ROW_ARG_DYN_2D:.*]]: index, %[[COL_ARG_DYN_2D:.*]]: index):
// CHECK: %[[C0_DYN_2D:.*]] = arith.constant 0 : index
// CHECK: %[[LOCAL_I_DYN_2D:.*]] = arith.subi %[[I_DYN_2D]], %[[ROW_ARG_DYN_2D]] : index
// CHECK: %[[LOCAL_J_DYN_2D:.*]] = arith.subi %[[J_DYN_2D]], %[[COL_ARG_DYN_2D]] : index
// CHECK: %[[VALUE_DYN_2D:.*]] = memref.load %[[VIEW_DYN_2D]][%[[LOCAL_I_DYN_2D]], %[[LOCAL_J_DYN_2D]]]
// CHECK: %[[TMP_DYN_2D:.*]] = memref.alloca() : memref<1xi32>
// CHECK: memref.store %[[VALUE_DYN_2D]], %[[TMP_DYN_2D]][%[[C0_DYN_2D]]] : memref<1xi32>
// CHECK: codir.yield
