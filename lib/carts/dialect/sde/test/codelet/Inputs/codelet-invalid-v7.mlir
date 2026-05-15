// Auxiliary input for ops-codelet-invalid.mlir (V7 check).
// The token slice type is memref<4xi32>, but the codelet block argument is
// declared as the unsliced source type.
module {
  func.func @v7_block_argument_type_mismatch(%m: memref<8xi32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %token = sde.mu_token <readwrite> %m [%c0] size [%c4]
      : memref<8xi32> -> !sde.token<memref<4xi32>>
    sde.cu_codelet (%token : !sde.token<memref<4xi32>>) {
    ^bb0(%arg: memref<8xi32>):
      sde.yield
    }
    func.return
  }
}
