// Auxiliary input for ops-codelet-invalid.mlir (V7 check).
// V7: codelet result count must equal the number of writable tokens. Here
//     we declare one <readwrite> token (1 writable) but request zero results.
module {
  func.func @v7_result_count_mismatch(%t: tensor<8xi32>) -> tensor<8xi32> {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %token = sde.mu_token <readwrite> %t [%c0] size [%c4]
      : tensor<8xi32> -> !sde.token<tensor<4xi32>>
    // Missing result; V7 requires exactly one (matching the readwrite token).
    sde.cu_codelet (%token : !sde.token<tensor<4xi32>>) {
    ^bb0(%arg: tensor<4xi32>):
      sde.yield
    }
    func.return %t : tensor<8xi32>
  }
}
