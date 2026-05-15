// Invalid-IR tests for the memref-native SDE codelet surface. Each RUN line
// invokes `carts-compile` on IR that violates a verifier rule; `not` flips the
// exit code so the run succeeds iff compilation fails.
//
// The rules exercised here are:
//   V2  — rank mismatch between `mu_token` offsets/sizes and the source rank.
//   V7  — `cu_codelet` block arguments must match token slice types.
//   V10 — memref codelets must not yield replacement values.
//   V11 — `cu_codelet` captures must be scalar values.
//
// RUN: not %carts-compile %s --arts-config %arts_config --pipeline initial-cleanup --start-from initial-cleanup 2>&1 | %FileCheck %s --check-prefix=V2
// RUN: not %carts-compile %S/Inputs/codelet-invalid-v7.mlir --arts-config %arts_config --pipeline initial-cleanup --start-from initial-cleanup 2>&1 | %FileCheck %s --check-prefix=V7
// RUN: not %carts-compile %S/Inputs/codelet-invalid-v10.mlir --arts-config %arts_config --pipeline initial-cleanup --start-from initial-cleanup 2>&1 | %FileCheck %s --check-prefix=V10
// RUN: not %carts-compile %S/Inputs/codelet-invalid-v11.mlir --arts-config %arts_config --pipeline initial-cleanup --start-from initial-cleanup 2>&1 | %FileCheck %s --check-prefix=V11

// V2: 'sde.mu_token' op expects offsets/sizes count (2) to match source rank (1)

// V7: 'sde.cu_codelet' op block argument #0 type ('memref<8xi32>') does not match token slice type ('memref<4xi32>')

// V10: 'sde.cu_codelet' op expects memref codelet yield to carry no values

// V11: 'sde.cu_codelet' op capture operand #0 must be an integer, index, or float scalar

module {
  func.func @v2_rank_mismatch(%m: memref<8xi32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %token = sde.mu_token <read> %m [%c0, %c0] size [%c4, %c4]
      : memref<8xi32> -> !sde.token<memref<4xi32>>
    func.return
  }
}
