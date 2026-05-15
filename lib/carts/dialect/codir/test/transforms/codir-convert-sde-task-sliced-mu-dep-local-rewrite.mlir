// RUN: %carts-compile %s --pass-pipeline='builtin.module(convert-sde-to-codir,verify-codir)' | %FileCheck %s --implicit-check-not=sde.cu_task --implicit-check-not=sde.mu_dep

module {
  func.func @task_sliced_mu_dep_rewrites_direct_source_access(%arg0: memref<16xi32>, %i: index) {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %dep = sde.mu_dep <readwrite> %arg0[%c4] size[%c8]
      : memref<16xi32> -> !sde.dep

    sde.cu_task deps(%dep : !sde.dep) {
      %v = memref.load %arg0[%i] : memref<16xi32>
      memref.store %v, %arg0[%i] : memref<16xi32>
      sde.yield
    }
    return
  }

  func.func @task_sliced_mu_dep_rewrites_body_subview(%arg0: memref<16xi32>, %i: index) {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %dep = sde.mu_dep <readwrite> %arg0[%c4] size[%c8]
      : memref<16xi32> -> !sde.dep

    sde.cu_task deps(%dep : !sde.dep) {
      %c0 = arith.constant 0 : index
      %view = memref.subview %arg0[%i] [1] [1]
        : memref<16xi32> to memref<1xi32, strided<[1], offset: ?>>
      %v = memref.load %view[%c0] : memref<1xi32, strided<[1], offset: ?>>
      memref.store %v, %view[%c0] : memref<1xi32, strided<[1], offset: ?>>
      sde.yield
    }
    return
  }

  func.func @task_sliced_mu_dep_rewrites_static_body_subview(%arg0: memref<16xi32>) {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %dep = sde.mu_dep <readwrite> %arg0[%c4] size[%c8]
      : memref<16xi32> -> !sde.dep

    sde.cu_task deps(%dep : !sde.dep) {
      %c0 = arith.constant 0 : index
      %view = memref.subview %arg0[5] [1] [1]
        : memref<16xi32> to memref<1xi32, strided<[1], offset: 5>>
      %v = memref.load %view[%c0] : memref<1xi32, strided<[1], offset: 5>>
      memref.store %v, %view[%c0] : memref<1xi32, strided<[1], offset: 5>>
      sde.yield
    }
    return
  }

  func.func @task_sliced_mu_dep_rewrites_body_subindex(%arg0: memref<16x16xi32>, %i: index, %j: index) {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %dep = sde.mu_dep <readwrite> %arg0[%c4, %c0] size[%c8, %c16]
      : memref<16x16xi32> -> !sde.dep

    sde.cu_task deps(%dep : !sde.dep) {
      %row = polygeist.subindex %arg0[%i] () : memref<16x16xi32> -> memref<16xi32>
      %v = memref.load %row[%j] : memref<16xi32>
      memref.store %v, %row[%j] : memref<16xi32>
      sde.yield
    }
    return
  }

  func.func @task_duplicate_same_source_mu_deps_reuses_sliced_view(%arg0: memref<16xi32>, %i: index) {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %read = sde.mu_dep <read> %arg0[%c4] size[%c8]
      : memref<16xi32> -> !sde.dep
    %write = sde.mu_dep <write> %arg0[%c4] size[%c8]
      : memref<16xi32> -> !sde.dep

    sde.cu_task deps(%read, %write : !sde.dep, !sde.dep) {
      %v = memref.load %arg0[%i] : memref<16xi32>
      memref.store %v, %arg0[%i] : memref<16xi32>
      sde.yield
    }
    return
  }

  func.func @task_duplicate_same_source_mixed_mu_dep_slices_use_positional_views(%arg0: memref<16xi32>, %i: index) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %read = sde.mu_dep <read> %arg0[%c0] size[%c8]
      : memref<16xi32> -> !sde.dep
    %write = sde.mu_dep <write> %arg0[%c8] size[%c8]
      : memref<16xi32> -> !sde.dep

    sde.cu_task deps(%read, %write : !sde.dep, !sde.dep) {
      %in = memref.subview %arg0[0] [8] [1]
        : memref<16xi32> to memref<8xi32, strided<[1]>>
      %out = memref.subview %arg0[8] [8] [1]
        : memref<16xi32> to memref<8xi32, strided<[1], offset: 8>>
      %v = memref.load %in[%i] : memref<8xi32, strided<[1]>>
      memref.store %v, %out[%i] : memref<8xi32, strided<[1], offset: 8>>
      sde.yield
    }
    return
  }

  func.func @task_duplicate_same_source_mixed_mu_dep_subindex_rows_use_positional_views(%arg0: memref<16x16xi32>, %j: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %read = sde.mu_dep <read> %arg0[%c0, %c0] size[%c1, %c16]
      : memref<16x16xi32> -> !sde.dep
    %write = sde.mu_dep <write> %arg0[%c1, %c0] size[%c1, %c16]
      : memref<16x16xi32> -> !sde.dep

    sde.cu_task deps(%read, %write : !sde.dep, !sde.dep) {
      %in = polygeist.subindex %arg0[%c0] () : memref<16x16xi32> -> memref<16xi32>
      %out = polygeist.subindex %arg0[%c1] () : memref<16x16xi32> -> memref<16xi32>
      %v = memref.load %in[%j] : memref<16xi32>
      memref.store %v, %out[%j] : memref<16xi32>
      sde.yield
    }
    return
  }

  func.func @task_duplicate_same_source_mixed_mu_dep_direct_root_access_not_sliced(%arg0: memref<16xi32>, %i: index) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %read = sde.mu_dep <read> %arg0[%c0] size[%c8]
      : memref<16xi32> -> !sde.dep
    %write = sde.mu_dep <write> %arg0[%c8] size[%c8]
      : memref<16xi32> -> !sde.dep

    sde.cu_task deps(%read, %write : !sde.dep, !sde.dep) {
      %v = memref.load %arg0[%i] : memref<16xi32>
      memref.store %v, %arg0[%i] : memref<16xi32>
      sde.yield
    }
    return
  }

  func.func @task_duplicate_same_source_mixed_mu_dep_subindex_mismatch_not_sliced(%arg0: memref<16x16xi32>, %i: index, %j: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %read = sde.mu_dep <read> %arg0[%c0, %c0] size[%c1, %c16]
      : memref<16x16xi32> -> !sde.dep
    %write = sde.mu_dep <write> %arg0[%c1, %c0] size[%c1, %c16]
      : memref<16x16xi32> -> !sde.dep

    sde.cu_task deps(%read, %write : !sde.dep, !sde.dep) {
      %row = polygeist.subindex %arg0[%i] () : memref<16x16xi32> -> memref<16xi32>
      %v = memref.load %row[%j] : memref<16xi32>
      memref.store %v, %row[%j] : memref<16xi32>
      sde.yield
    }
    return
  }

  func.func @task_duplicate_same_source_mixed_mu_dep_direct_root_exact_offsets_sliced(%arg0: memref<16xi32>) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %read = sde.mu_dep <read> %arg0[%c0] size[%c8]
      : memref<16xi32> -> !sde.dep
    %write = sde.mu_dep <write> %arg0[%c8] size[%c8]
      : memref<16xi32> -> !sde.dep

    sde.cu_task deps(%read, %write : !sde.dep, !sde.dep) {
      %v = memref.load %arg0[%c0] : memref<16xi32>
      memref.store %v, %arg0[%c8] : memref<16xi32>
      sde.yield
    }
    return
  }

  func.func @task_dynamic_mu_dep_exact_body_subview(%arg0: memref<?xi32>, %off: index, %n: index) {
    %dep = sde.mu_dep <readwrite> %arg0[%off] size[%n]
      : memref<?xi32> -> !sde.dep

    sde.cu_task deps(%dep : !sde.dep) {
      %c0 = arith.constant 0 : index
      %view = memref.subview %arg0[%off] [%n] [1]
        : memref<?xi32> to memref<?xi32, strided<[1], offset: ?>>
      %v = memref.load %view[%c0] : memref<?xi32, strided<[1], offset: ?>>
      memref.store %v, %view[%c0] : memref<?xi32, strided<[1], offset: ?>>
      sde.yield
    }
    return
  }

  func.func @task_dynamic_mu_dep_exact_root_access_sliced(%arg0: memref<?xi32>, %off: index, %n: index) {
    %dep = sde.mu_dep <readwrite> %arg0[%off] size[%n]
      : memref<?xi32> -> !sde.dep

    sde.cu_task deps(%dep : !sde.dep) {
      %v = memref.load %arg0[%off] : memref<?xi32>
      memref.store %v, %arg0[%off] : memref<?xi32>
      sde.yield
    }
    return
  }

  func.func @task_dynamic_mu_dep_direct_root_access_not_sliced(%arg0: memref<?xi32>, %off: index, %i: index, %n: index) {
    %dep = sde.mu_dep <readwrite> %arg0[%off] size[%n]
      : memref<?xi32> -> !sde.dep

    sde.cu_task deps(%dep : !sde.dep) {
      %v = memref.load %arg0[%i] : memref<?xi32>
      memref.store %v, %arg0[%i] : memref<?xi32>
      sde.yield
    }
    return
  }

  func.func @task_dynamic_mu_dep_mismatched_body_subview_not_sliced(%arg0: memref<?xi32>, %off: index, %other: index, %n: index) {
    %dep = sde.mu_dep <readwrite> %arg0[%off] size[%n]
      : memref<?xi32> -> !sde.dep

    sde.cu_task deps(%dep : !sde.dep) {
      %c0 = arith.constant 0 : index
      %view = memref.subview %arg0[%other] [%n] [1]
        : memref<?xi32> to memref<?xi32, strided<[1], offset: ?>>
      %v = memref.load %view[%c0] : memref<?xi32, strided<[1], offset: ?>>
      memref.store %v, %view[%c0] : memref<?xi32, strided<[1], offset: ?>>
      sde.yield
    }
    return
  }

}

// CHECK-LABEL: func.func @task_sliced_mu_dep_rewrites_direct_source_access
// CHECK: %[[C4_OUT:.*]] = arith.constant 4 : index
// CHECK: %[[C8:.*]] = arith.constant 8 : index
// CHECK: %[[VIEW:.*]] = memref.subview %arg0[%[[C4_OUT]]] [%[[C8]]] [1]
// CHECK: codir.codelet deps(%[[VIEW]] :
// CHECK-SAME: params(%arg1 : index)
// CHECK-SAME: dep_modes = [#codir.access_mode<readwrite>]
// CHECK-SAME: ordered_task_depend
// CHECK-SAME: task_depend
// CHECK: ^bb0(%[[DEP:[a-zA-Z0-9_]+]]: {{.*}}, %[[I_ARG:[a-zA-Z0-9_]+]]: index):
// CHECK: %[[C4_IN:.*]] = arith.constant 4 : index
// CHECK: %[[LOCAL_LOAD:.*]] = arith.subi %[[I_ARG]], %[[C4_IN]] : index
// CHECK: %[[V:.*]] = memref.load %[[DEP]][%[[LOCAL_LOAD]]]
// CHECK: %[[LOCAL_STORE:.*]] = arith.subi %[[I_ARG]], %[[C4_IN]] : index
// CHECK: memref.store %[[V]], %[[DEP]][%[[LOCAL_STORE]]]
// CHECK: codir.yield

// CHECK-LABEL: func.func @task_sliced_mu_dep_rewrites_body_subview
// CHECK: %[[C4_OUT:.*]] = arith.constant 4 : index
// CHECK: %[[C8_OUT:.*]] = arith.constant 8 : index
// CHECK: %[[DEP_VIEW:.*]] = memref.subview %arg0[%[[C4_OUT]]] [%[[C8_OUT]]] [1]
// CHECK: codir.codelet deps(%[[DEP_VIEW]] :
// CHECK-SAME: params(%arg1 : index)
// CHECK: ^bb0(%[[DEP_ARG:[a-zA-Z0-9_]+]]: {{.*}}, %[[I_ARG:[a-zA-Z0-9_]+]]: index):
// CHECK: %[[C4_IN:.*]] = arith.constant 4 : index
// CHECK: %[[LOCAL_VIEW_OFFSET:.*]] = arith.subi %[[I_ARG]], %[[C4_IN]] : index
// CHECK: %[[LOCAL_VIEW:.*]] = memref.subview %[[DEP_ARG]][%[[LOCAL_VIEW_OFFSET]]] [1] [1]
// CHECK: %[[VIEW_LOAD:.*]] = memref.load %[[LOCAL_VIEW]][
// CHECK: memref.store %[[VIEW_LOAD]], %[[LOCAL_VIEW]][
// CHECK: codir.yield

// CHECK-LABEL: func.func @task_sliced_mu_dep_rewrites_static_body_subview
// CHECK: %[[C4_STATIC_OUT:.*]] = arith.constant 4 : index
// CHECK: %[[C8_STATIC_OUT:.*]] = arith.constant 8 : index
// CHECK: %[[DEP_STATIC_VIEW:.*]] = memref.subview %arg0[%[[C4_STATIC_OUT]]] [%[[C8_STATIC_OUT]]] [1]
// CHECK: codir.codelet deps(%[[DEP_STATIC_VIEW]] :
// CHECK: ^bb0(%[[DEP_STATIC_ARG:[a-zA-Z0-9_]+]]: {{.*}}):
// CHECK: %[[LOCAL_STATIC_VIEW:.*]] = memref.subview %[[DEP_STATIC_ARG]][1] [1] [1]
// CHECK-SAME: memref<?xi32, strided<[1], offset: ?>> to memref<1xi32, strided<[1], offset: ?>>
// CHECK: memref.load %[[LOCAL_STATIC_VIEW]]
// CHECK: memref.store {{.*}}, %[[LOCAL_STATIC_VIEW]]
// CHECK: codir.yield

// CHECK-LABEL: func.func @task_sliced_mu_dep_rewrites_body_subindex
// CHECK: %[[C4_2D_OUT:.*]] = arith.constant 4 : index
// CHECK: %[[C8_2D_OUT:.*]] = arith.constant 8 : index
// CHECK: %[[C0_2D_OUT:.*]] = arith.constant 0 : index
// CHECK: %[[C16_2D_OUT:.*]] = arith.constant 16 : index
// CHECK: %[[DEP_VIEW_2D:.*]] = memref.subview %arg0[%[[C4_2D_OUT]], %[[C0_2D_OUT]]] [%[[C8_2D_OUT]], %[[C16_2D_OUT]]] [1, 1]
// CHECK: codir.codelet deps(%[[DEP_VIEW_2D]] :
// CHECK-SAME: params(%arg1, %arg2 : index, index)
// CHECK: ^bb0(%[[DEP_ARG_2D:[a-zA-Z0-9_]+]]: {{.*}}, %[[I_ARG_2D:[a-zA-Z0-9_]+]]: index, %[[J_ARG_2D:[a-zA-Z0-9_]+]]: index):
// CHECK: %[[C4_2D_IN:.*]] = arith.constant 4 : index
// CHECK: %[[LOCAL_SUBINDEX:.*]] = arith.subi %[[I_ARG_2D]], %[[C4_2D_IN]] : index
// CHECK: %[[ROW:.*]] = polygeist.subindex %[[DEP_ARG_2D]][%[[LOCAL_SUBINDEX]]]
// CHECK: %[[ROW_LOAD:.*]] = memref.load %[[ROW]][%[[J_ARG_2D]]]
// CHECK: memref.store %[[ROW_LOAD]], %[[ROW]][%[[J_ARG_2D]]]
// CHECK: codir.yield

// CHECK-LABEL: func.func @task_duplicate_same_source_mu_deps_reuses_sliced_view
// CHECK: %[[C4_DUP_OUT:.*]] = arith.constant 4 : index
// CHECK: %[[C8_DUP_OUT:.*]] = arith.constant 8 : index
// CHECK: %[[DUP_VIEW:.*]] = memref.subview %arg0[%[[C4_DUP_OUT]]] [%[[C8_DUP_OUT]]] [1]
// CHECK: codir.codelet deps(%[[DUP_VIEW]] :
// CHECK-SAME: params(%arg1 : index)
// CHECK-SAME: dep_modes = [#codir.access_mode<readwrite>]
// CHECK: ^bb0(%[[DUP_DEP:[a-zA-Z0-9_]+]]: {{.*}}, %[[DUP_I:[a-zA-Z0-9_]+]]: index):
// CHECK: %[[C4_DUP_IN:.*]] = arith.constant 4 : index
// CHECK: %[[DUP_LOCAL_LOAD:.*]] = arith.subi %[[DUP_I]], %[[C4_DUP_IN]] : index
// CHECK: %[[DUP_V:.*]] = memref.load %[[DUP_DEP]][%[[DUP_LOCAL_LOAD]]]
// CHECK: %[[DUP_LOCAL_STORE:.*]] = arith.subi %[[DUP_I]], %[[C4_DUP_IN]] : index
// CHECK: memref.store %[[DUP_V]], %[[DUP_DEP]][%[[DUP_LOCAL_STORE]]]
// CHECK: codir.yield

// CHECK-LABEL: func.func @task_duplicate_same_source_mixed_mu_dep_slices_use_positional_views
// CHECK: %[[C0_MIXED_OUT:.*]] = arith.constant 0 : index
// CHECK: %[[C8_MIXED_OUT:.*]] = arith.constant 8 : index
// CHECK: %[[READ_VIEW:.*]] = memref.subview %arg0[%[[C0_MIXED_OUT]]] [%[[C8_MIXED_OUT]]] [1]
// CHECK: %[[WRITE_VIEW:.*]] = memref.subview %arg0[%[[C8_MIXED_OUT]]] [%[[C8_MIXED_OUT]]] [1]
// CHECK: codir.codelet deps(%[[READ_VIEW]], %[[WRITE_VIEW]] :
// CHECK-SAME: params(%arg1 : index)
// CHECK-SAME: dep_modes = [#codir.access_mode<read>, #codir.access_mode<write>]
// CHECK: ^bb0(%[[READ_DEP:[a-zA-Z0-9_]+]]: {{.*}}, %[[WRITE_DEP:[a-zA-Z0-9_]+]]: {{.*}}, %[[MIXED_I:[a-zA-Z0-9_]+]]: index):
// CHECK: %[[MIXED_V:.*]] = memref.load %[[READ_DEP]][%[[MIXED_I]]]
// CHECK: memref.store %[[MIXED_V]], %[[WRITE_DEP]][%[[MIXED_I]]]
// CHECK: codir.yield

// CHECK-LABEL: func.func @task_duplicate_same_source_mixed_mu_dep_subindex_rows_use_positional_views
// CHECK: %[[C0_ROW_OUT:.*]] = arith.constant 0 : index
// CHECK: %[[C1_ROW_OUT:.*]] = arith.constant 1 : index
// CHECK: %[[C16_ROW_OUT:.*]] = arith.constant 16 : index
// CHECK: %[[READ_ROW_VIEW:.*]] = memref.subview %arg0[%[[C0_ROW_OUT]], %[[C0_ROW_OUT]]] [%[[C1_ROW_OUT]], %[[C16_ROW_OUT]]] [1, 1]
// CHECK: %[[WRITE_ROW_VIEW:.*]] = memref.subview %arg0[%[[C1_ROW_OUT]], %[[C0_ROW_OUT]]] [%[[C1_ROW_OUT]], %[[C16_ROW_OUT]]] [1, 1]
// CHECK: codir.codelet deps(%[[READ_ROW_VIEW]], %[[WRITE_ROW_VIEW]] :
// CHECK-SAME: params(%arg1 : index)
// CHECK-SAME: dep_modes = [#codir.access_mode<read>, #codir.access_mode<write>]
// CHECK: ^bb0(%[[READ_ROW_DEP:[a-zA-Z0-9_]+]]: {{.*}}, %[[WRITE_ROW_DEP:[a-zA-Z0-9_]+]]: {{.*}}, %[[ROW_J:[a-zA-Z0-9_]+]]: index):
// CHECK: %[[READ_ROW_ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[LOCAL_IN:.*]] = polygeist.subindex %[[READ_ROW_DEP]][%[[READ_ROW_ZERO]]] ()
// CHECK: %[[WRITE_ROW_ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[LOCAL_OUT:.*]] = polygeist.subindex %[[WRITE_ROW_DEP]][%[[WRITE_ROW_ZERO]]] ()
// CHECK: %[[ROW_V:.*]] = memref.load %[[LOCAL_IN]][%[[ROW_J]]]
// CHECK: memref.store %[[ROW_V]], %[[LOCAL_OUT]][%[[ROW_J]]]
// CHECK: codir.yield

// CHECK-LABEL: func.func @task_duplicate_same_source_mixed_mu_dep_direct_root_access_not_sliced
// CHECK-NOT: memref.subview %arg0[
// CHECK: codir.codelet deps(%arg0 : memref<16xi32>)
// CHECK-SAME: params(%arg1 : index)
// CHECK-SAME: dep_modes = [#codir.access_mode<readwrite>]
// CHECK: ^bb0(%[[MIXED_ROOT_DEP:[a-zA-Z0-9_]+]]: memref<16xi32>, %[[MIXED_ROOT_I:[a-zA-Z0-9_]+]]: index):
// CHECK: %[[MIXED_ROOT_V:.*]] = memref.load %[[MIXED_ROOT_DEP]][%[[MIXED_ROOT_I]]]
// CHECK: memref.store %[[MIXED_ROOT_V]], %[[MIXED_ROOT_DEP]][%[[MIXED_ROOT_I]]]
// CHECK: codir.yield

// CHECK-LABEL: func.func @task_duplicate_same_source_mixed_mu_dep_subindex_mismatch_not_sliced
// CHECK-NOT: memref.subview %arg0[
// CHECK: codir.codelet deps(%arg0 : memref<16x16xi32>)
// CHECK-SAME: params(%arg1, %arg2 : index, index)
// CHECK-SAME: dep_modes = [#codir.access_mode<readwrite>]
// CHECK: ^bb0(%[[MISMATCH_ROW_DEP:[a-zA-Z0-9_]+]]: memref<16x16xi32>, %[[MISMATCH_ROW_I:[a-zA-Z0-9_]+]]: index, %[[MISMATCH_ROW_J:[a-zA-Z0-9_]+]]: index):
// CHECK: %[[MISMATCH_ROW:.*]] = polygeist.subindex %[[MISMATCH_ROW_DEP]][%[[MISMATCH_ROW_I]]] ()
// CHECK: %[[MISMATCH_ROW_V:.*]] = memref.load %[[MISMATCH_ROW]][%[[MISMATCH_ROW_J]]]
// CHECK: memref.store %[[MISMATCH_ROW_V]], %[[MISMATCH_ROW]][%[[MISMATCH_ROW_J]]]
// CHECK: codir.yield

// CHECK-LABEL: func.func @task_duplicate_same_source_mixed_mu_dep_direct_root_exact_offsets_sliced
// CHECK: %[[C0_MIXED_ROOT_OUT:.*]] = arith.constant 0 : index
// CHECK: %[[C8_MIXED_ROOT_OUT:.*]] = arith.constant 8 : index
// CHECK: %[[READ_ROOT_VIEW:.*]] = memref.subview %arg0[%[[C0_MIXED_ROOT_OUT]]] [%[[C8_MIXED_ROOT_OUT]]] [1]
// CHECK: %[[WRITE_ROOT_VIEW:.*]] = memref.subview %arg0[%[[C8_MIXED_ROOT_OUT]]] [%[[C8_MIXED_ROOT_OUT]]] [1]
// CHECK: codir.codelet deps(%[[READ_ROOT_VIEW]], %[[WRITE_ROOT_VIEW]] :
// CHECK-SAME: dep_modes = [#codir.access_mode<read>, #codir.access_mode<write>]
// CHECK: ^bb0(%[[READ_ROOT_DEP:[a-zA-Z0-9_]+]]: {{.*}}, %[[WRITE_ROOT_DEP:[a-zA-Z0-9_]+]]: {{.*}}):
// CHECK: %[[READ_ROOT_ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[MIXED_ROOT_EXACT_V:.*]] = memref.load %[[READ_ROOT_DEP]][%[[READ_ROOT_ZERO]]]
// CHECK: memref.store %[[MIXED_ROOT_EXACT_V]], %[[WRITE_ROOT_DEP]][%{{.*}}]
// CHECK: codir.yield

// CHECK-LABEL: func.func @task_dynamic_mu_dep_exact_body_subview
// CHECK: %[[DYN_DEP_VIEW:.*]] = memref.subview %arg0[%arg1] [%arg2] [1]
// CHECK: codir.codelet deps(%[[DYN_DEP_VIEW]] :
// CHECK-SAME: params(%arg1, %arg2 : index, index)
// CHECK-SAME: dep_modes = [#codir.access_mode<readwrite>]
// CHECK: ^bb0(%[[DYN_DEP:[a-zA-Z0-9_]+]]: {{.*}}, %[[DYN_OFF:[a-zA-Z0-9_]+]]: index, %[[DYN_N:[a-zA-Z0-9_]+]]: index):
// CHECK: arith.constant 0 : index
// CHECK: %[[DYN_LOCAL_VIEW:.*]] = memref.subview %[[DYN_DEP]][%{{.*}}] [%[[DYN_N]]] [1]
// CHECK: %[[DYN_V:.*]] = memref.load %[[DYN_LOCAL_VIEW]]
// CHECK: memref.store %[[DYN_V]], %[[DYN_LOCAL_VIEW]]
// CHECK: codir.yield

// CHECK-LABEL: func.func @task_dynamic_mu_dep_exact_root_access_sliced
// CHECK: %[[DYN_ROOT_VIEW:.*]] = memref.subview %arg0[%arg1] [%arg2] [1]
// CHECK: codir.codelet deps(%[[DYN_ROOT_VIEW]] :
// CHECK-SAME: params(%arg1, %arg2 : index, index)
// CHECK-SAME: dep_modes = [#codir.access_mode<readwrite>]
// CHECK: ^bb0(%[[DYN_ROOT_DEP:[a-zA-Z0-9_]+]]: {{.*}}, %[[DYN_ROOT_OFF:[a-zA-Z0-9_]+]]: index, %[[DYN_ROOT_N:[a-zA-Z0-9_]+]]: index):
// CHECK: %[[DYN_ROOT_ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[DYN_ROOT_V:.*]] = memref.load %[[DYN_ROOT_DEP]][%[[DYN_ROOT_ZERO]]]
// CHECK: memref.store %[[DYN_ROOT_V]], %[[DYN_ROOT_DEP]][%{{.*}}]
// CHECK: codir.yield

// CHECK-LABEL: func.func @task_dynamic_mu_dep_direct_root_access_not_sliced
// CHECK-NOT: memref.subview %arg0[%arg1] [%arg3] [1]
// CHECK: codir.codelet deps(%arg0 :
// CHECK-SAME: params(%arg2 : index)
// CHECK: ^bb0(%[[ROOT_DEP:[a-zA-Z0-9_]+]]: memref<?xi32>, %[[ROOT_I:[a-zA-Z0-9_]+]]: index):
// CHECK: memref.load %[[ROOT_DEP]][%[[ROOT_I]]]
// CHECK: memref.store {{.*}}, %[[ROOT_DEP]][%[[ROOT_I]]]
// CHECK: codir.yield

// CHECK-LABEL: func.func @task_dynamic_mu_dep_mismatched_body_subview_not_sliced
// CHECK-NOT: memref.subview %arg0[%arg1] [%arg3] [1]
// CHECK: codir.codelet deps(%arg0 :
// CHECK-SAME: params(%arg2, %arg3 : index, index)
// CHECK: ^bb0(%[[MISMATCH_DEP:[a-zA-Z0-9_]+]]: memref<?xi32>, %[[MISMATCH_OTHER:[a-zA-Z0-9_]+]]: index, %[[MISMATCH_N:[a-zA-Z0-9_]+]]: index):
// CHECK: %[[MISMATCH_VIEW:.*]] = memref.subview %[[MISMATCH_DEP]][%[[MISMATCH_OTHER]]] [%[[MISMATCH_N]]] [1]
// CHECK: memref.load %[[MISMATCH_VIEW]]
// CHECK: memref.store {{.*}}, %[[MISMATCH_VIEW]]
// CHECK: codir.yield
