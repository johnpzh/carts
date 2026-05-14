// Invalid-IR tests for the RFC step-1 SDE ops. Each RUN line invokes
// `carts-compile` on IR that violates a specific verifier rule; `not` flips
// the exit code so the run succeeds iff compilation fails.
//
// The rules exercised here are:
//   V2  — rank mismatch between `mu_token` offsets/sizes and the source
//         tensor rank.
//   V7  — `cu_codelet` result count must equal the number of writable
//         tokens (write / readwrite).
//   V10 — a <read> token cannot have a yielded counterpart; enforced
//         through the same V7 gate, whose diagnostic explicitly cites V10.
//   V11 — `cu_codelet` captures must be scalar values.
//
// RUN: not %carts-compile %s --arts-config %arts_config --pipeline initial-cleanup --start-from initial-cleanup 2>&1 | %FileCheck %s --check-prefix=V2
// RUN: not %carts-compile %S/Inputs/codelet-invalid-v7.mlir --arts-config %arts_config --pipeline initial-cleanup --start-from initial-cleanup 2>&1 | %FileCheck %s --check-prefix=V7
// RUN: not %carts-compile %S/Inputs/codelet-invalid-v10.mlir --arts-config %arts_config --pipeline initial-cleanup --start-from initial-cleanup 2>&1 | %FileCheck %s --check-prefix=V10
// RUN: not %carts-compile %S/Inputs/codelet-invalid-v11.mlir --arts-config %arts_config --pipeline initial-cleanup --start-from initial-cleanup 2>&1 | %FileCheck %s --check-prefix=V11

// V2: 'arts_sde.mu_token' op expects offsets/sizes count (2) to match source tensor rank (1)

// V7: 'arts_sde.cu_codelet' op expects one result per writable token
// V7-SAME: got 0 result(s) and 1 writable

// V10: 'arts_sde.cu_codelet' op expects one result per writable token
// V10-SAME: {{<read> token must not have a yielded counterpart}}

// V11: 'arts_sde.cu_codelet' op capture operand #0 must be an integer, index, or float scalar

module {
  func.func @v2_rank_mismatch(%t: tensor<8xi32>) -> tensor<8xi32> {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %token = arts_sde.mu_token <read> %t [%c0, %c0] size [%c4, %c4]
      : tensor<8xi32> -> !arts_sde.token<tensor<4xi32>>
    func.return %t : tensor<8xi32>
  }
}
