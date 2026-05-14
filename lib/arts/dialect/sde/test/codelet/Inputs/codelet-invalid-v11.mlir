module {
  func.func @v11_memref_capture(%t: tensor<4xi32>, %m: memref<4xi32>) {
    %token = arts_sde.mu_token <read> %t
      : tensor<4xi32> -> !arts_sde.token<tensor<4xi32>>
    arts_sde.cu_codelet (%token : !arts_sde.token<tensor<4xi32>>)
      captures(%m : memref<4xi32>) {
    ^bb0(%view: tensor<4xi32>, %capture: memref<4xi32>):
      arts_sde.yield
    }
    return
  }
}
