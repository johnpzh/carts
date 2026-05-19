// RUN: %carts-compile %s --pass-pipeline='builtin.module(convert-sde-to-codir,verify-codir)' | %FileCheck %s --implicit-check-not=sde.cu_codelet

module {
  func.func @sde_explicit_storage_view_to_codir(%mem: memref<8xf32>) {
    %token = sde.mu_token <read> %mem
      : memref<8xf32> -> !sde.token<memref<8xf32>>
      {storageView = #sde.storage_view<replicated_read>}

    sde.cu_codelet (%token : !sde.token<memref<8xf32>>) {
    ^bb0(%view: memref<8xf32>):
      %c0 = arith.constant 0 : index
      %value = memref.load %view[%c0] : memref<8xf32>
      sde.yield
    }
    return
  }
}

// CHECK-LABEL: func.func @sde_explicit_storage_view_to_codir
// CHECK: sde.mu_token
// CHECK-SAME: storageView = #sde.storage_view<replicated_read>
// CHECK: codir.codelet deps(%{{.*}} : memref<8xf32>)
// CHECK-SAME: dep_modes = [#codir.access_mode<read>]
// CHECK-SAME: dep_storage_views = [#codir.storage_view<replicated_read>]
