module {
  func.func @v11_memref_capture(%m: memref<4xi32>, %capture: memref<4xi32>) {
    %token = sde.mu_token <read> %m
      : memref<4xi32> -> !sde.token<memref<4xi32>>
    sde.cu_codelet (%token : !sde.token<memref<4xi32>>)
      captures(%capture : memref<4xi32>) {
    ^bb0(%view: memref<4xi32>, %captured: memref<4xi32>):
      sde.yield
    }
    return
  }
}
