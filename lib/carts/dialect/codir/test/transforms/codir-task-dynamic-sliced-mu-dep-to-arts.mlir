// RUN: %carts-compile %s --pass-pipeline='builtin.module(convert-sde-to-codir,verify-codir,convert-codir-to-arts)' | %FileCheck %s --implicit-check-not=sde.cu_task --implicit-check-not=sde.mu_dep --implicit-check-not=codir.codelet

module {
  func.func @task_dynamic_mu_dep_exact_body_subview_to_arts(%arg0: index,
                                                           %off: index,
                                                           %n: index) {
    %root = sde.mu_alloc(%arg0) : memref<?xi32>
    %dep = sde.mu_dep <readwrite> %root[%off] size[%n]
      : memref<?xi32> -> !sde.dep

    sde.cu_task deps(%dep : !sde.dep) {
      %c0 = arith.constant 0 : index
      %view = memref.subview %root[%off] [%n] [1]
        : memref<?xi32> to memref<?xi32, strided<[1], offset: ?>>
      %v = memref.load %view[%c0] : memref<?xi32, strided<[1], offset: ?>>
      memref.store %v, %view[%c0] : memref<?xi32, strided<[1], offset: ?>>
      sde.yield
    }
    return
  }

  func.func @task_dynamic_mu_dep_exact_root_access_to_arts(%arg0: index,
                                                          %off: index,
                                                          %n: index) {
    %root = sde.mu_alloc(%arg0) : memref<?xi32>
    %dep = sde.mu_dep <readwrite> %root[%off] size[%n]
      : memref<?xi32> -> !sde.dep

    sde.cu_task deps(%dep : !sde.dep) {
      %v = memref.load %root[%off] : memref<?xi32>
      memref.store %v, %root[%off] : memref<?xi32>
      sde.yield
    }
    return
  }
}

// CHECK-LABEL: func.func @task_dynamic_mu_dep_exact_body_subview_to_arts
// CHECK: arts.db_acquire
// CHECK: arts.edt
// CHECK-SAME: params(%arg1, %arg2, %arg0 : index, index, index)
// CHECK: ^bb0(%[[DB:[a-zA-Z0-9_]+]]: {{.*}}, %[[OFF:[a-zA-Z0-9_]+]]: index, %[[N:[a-zA-Z0-9_]+]]: index, %[[ROOT_SIZE:[a-zA-Z0-9_]+]]: index):
// CHECK: %[[PAYLOAD:.*]] = arts.db_ref %[[DB]]
// CHECK: %[[VIEW:.*]] = memref.subview %[[PAYLOAD]][%[[OFF]]] [%[[N]]] [1]
// CHECK: %[[LOCAL:.*]] = memref.subview %[[VIEW]][%{{.*}}] [%[[N]]] [1]
// CHECK: %[[V:.*]] = memref.load %[[LOCAL]]
// CHECK: memref.store %[[V]], %[[LOCAL]]
// CHECK: arts.barrier

// CHECK-LABEL: func.func @task_dynamic_mu_dep_exact_root_access_to_arts
// CHECK: arts.db_acquire
// CHECK: arts.edt
// CHECK-SAME: params(%arg1, %arg2, %arg0 : index, index, index)
// CHECK: ^bb0(%[[ROOT_DB:[a-zA-Z0-9_]+]]: {{.*}}, %[[ROOT_OFF:[a-zA-Z0-9_]+]]: index, %[[ROOT_N:[a-zA-Z0-9_]+]]: index, %[[ROOT_SIZE:[a-zA-Z0-9_]+]]: index):
// CHECK: %[[ROOT_PAYLOAD:.*]] = arts.db_ref %[[ROOT_DB]]
// CHECK: %[[ROOT_VIEW:.*]] = memref.subview %[[ROOT_PAYLOAD]][%[[ROOT_OFF]]] [%[[ROOT_N]]] [1]
// CHECK: %[[ROOT_ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[ROOT_V:.*]] = memref.load %[[ROOT_VIEW]][%[[ROOT_ZERO]]]
// CHECK: memref.store %[[ROOT_V]], %[[ROOT_VIEW]][%{{.*}}]
// CHECK: arts.barrier
