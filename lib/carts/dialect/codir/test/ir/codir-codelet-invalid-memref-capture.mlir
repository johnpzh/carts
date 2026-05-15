// RUN: not %carts-compile %s --arts-config %arts_config --pipeline raise-memref-dimensionality 2>&1 | %FileCheck %s

// CHECK: 'memref.load' op using value defined outside the region

module {
  func.func @codir_rejects_implicit_memref_capture(%dep: memref<4xf32>) {
    codir.codelet {
      %c0 = arith.constant 0 : index
      %value = memref.load %dep[%c0] : memref<4xf32>
      codir.yield
    }
    return
  }
}
