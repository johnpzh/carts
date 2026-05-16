// RUN: %carts-compile %s --pass-pipeline='builtin.module(convert-sde-to-codir,verify-codir,convert-codir-to-arts)' | %FileCheck %s --implicit-check-not=codir.codelet --implicit-check-not=sde.

module {
  func.func @codir_to_arts_materializes_arts(%param: index) {
    %dep = sde.mu_alloc : memref<4xf32>
    codir.codelet deps(%dep : memref<4xf32>) params(%param : index) attributes {dep_modes = [#codir.access_mode<readwrite>]} {
    ^bb0(%dep_arg: memref<4xf32>, %param_arg: index):
      %c0 = arith.constant 0 : index
      %v = memref.load %dep_arg[%c0] : memref<4xf32>
      memref.store %v, %dep_arg[%c0] : memref<4xf32>
      codir.yield
    }
    return
  }

  func.func @sde_task_body_subview_reaches_arts(%i: index) {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %root = sde.mu_alloc : memref<16xi32>
    %dep = sde.mu_dep <readwrite> %root[%c4] size[%c8]
      : memref<16xi32> -> !sde.dep

    sde.cu_task deps(%dep : !sde.dep) {
      %c0 = arith.constant 0 : index
      %view = memref.subview %root[%i] [1] [1]
        : memref<16xi32> to memref<1xi32, strided<[1], offset: ?>>
      %v = memref.load %view[%c0] : memref<1xi32, strided<[1], offset: ?>>
      memref.store %v, %view[%c0] : memref<1xi32, strided<[1], offset: ?>>
      sde.yield
    }
    return
  }

  func.func @sde_task_body_subindex_reaches_arts(%i: index, %j: index) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %root = sde.mu_alloc : memref<16x16xi32>
    %dep = sde.mu_dep <readwrite> %root[%c4, %c0] size[%c8, %c16]
      : memref<16x16xi32> -> !sde.dep

    sde.cu_task deps(%dep : !sde.dep) {
      %row = polygeist.subindex %root[%i] () : memref<16x16xi32> -> memref<16xi32>
      %v = memref.load %row[%j] : memref<16xi32>
      memref.store %v, %row[%j] : memref<16xi32>
      sde.yield
    }
    return
  }

  func.func @sde_task_mixed_row_subindex_deps_reach_arts(%j: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %root = sde.mu_alloc : memref<16x16xi32>
    %read = sde.mu_dep <read> %root[%c0, %c0] size[%c1, %c16]
      : memref<16x16xi32> -> !sde.dep
    %write = sde.mu_dep <write> %root[%c1, %c0] size[%c1, %c16]
      : memref<16x16xi32> -> !sde.dep

    sde.cu_task deps(%read, %write : !sde.dep, !sde.dep) {
      %in = polygeist.subindex %root[%c0] () : memref<16x16xi32> -> memref<16xi32>
      %out = polygeist.subindex %root[%c1] () : memref<16x16xi32> -> memref<16xi32>
      %v = memref.load %in[%j] : memref<16xi32>
      memref.store %v, %out[%j] : memref<16xi32>
      sde.yield
    }
    return
  }

  func.func @codir_to_arts_anchors_nested_dispatch_barrier() {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %dep = sde.mu_alloc : memref<4x4xf32>
    scf.for %i = %c0 to %c4 step %c2 {
      scf.for %j = %c0 to %c4 step %c2 {
        codir.codelet deps(%dep : memref<4x4xf32>) params(%i, %j : index, index)
            attributes {completion_barrier, dep_modes = [#codir.access_mode<readwrite>]} {
        ^bb0(%dep_arg: memref<4x4xf32>, %ii: index, %jj: index):
          %v = memref.load %dep_arg[%ii, %jj] : memref<4x4xf32>
          memref.store %v, %dep_arg[%ii, %jj] : memref<4x4xf32>
          codir.yield
        }
      }
    }
    return
  }
}

// CHECK-LABEL: func.func @codir_to_arts_materializes_arts
// CHECK: arts.db_alloc
// CHECK: arts.db_acquire
// CHECK: arts.edt <task>
// CHECK: memref.load
// CHECK: memref.store
// CHECK: }

// CHECK-LABEL: func.func @sde_task_body_subview_reaches_arts
// CHECK: arts.db_alloc
// CHECK: arts.db_acquire
// CHECK: arts.edt <task>
// CHECK: arith.subi
// CHECK: memref.subview
// CHECK: memref.load
// CHECK: memref.store
// CHECK: }

// CHECK-LABEL: func.func @sde_task_body_subindex_reaches_arts
// CHECK: arts.db_alloc
// CHECK: arts.db_acquire
// CHECK: arts.edt <task>
// CHECK: arith.subi
// CHECK: polygeist.subindex
// CHECK: memref.load
// CHECK: memref.store
// CHECK: }

// CHECK-LABEL: func.func @sde_task_mixed_row_subindex_deps_reach_arts
// CHECK: arts.db_alloc
// CHECK: arts.db_acquire
// CHECK: arts.db_acquire
// CHECK: arts.edt <task>
// CHECK: polygeist.subindex
// CHECK: polygeist.subindex
// CHECK: memref.load
// CHECK: memref.store
// CHECK: }

// CHECK-LABEL: func.func @codir_to_arts_anchors_nested_dispatch_barrier
// CHECK: scf.for
// CHECK: scf.for
// CHECK: arts.edt <task>
// CHECK: }
// CHECK: }
// CHECK: arts.barrier
// CHECK: return
