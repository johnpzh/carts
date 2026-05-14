module {
  func.func @v11_memref_capture(%t: tensor<4xi32>, %m: memref<4xi32>) {
    %token = sde.mu_token <read> %t
      : tensor<4xi32> -> !sde.token<tensor<4xi32>>
    sde.cu_codelet (%token : !sde.token<tensor<4xi32>>)
      captures(%m : memref<4xi32>) {
    ^bb0(%view: tensor<4xi32>, %capture: memref<4xi32>):
      sde.yield
    }
    return
  }
}
