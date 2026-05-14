// RFC raise-memref-to-tensor step-2 lowering tests.
//
// ConvertSdeToArts must turn tensor-path SDE ops into ARTS core:
//   - sde.mu_data     : tensor<...>        -> arts.db_alloc
//   - sde.mu_token    <mode>                -> arts.db_acquire <mode>
//   - sde.cu_codelet (tokens...) -> (...)    -> arts.edt
//
// We run the full `openmp-to-arts` stage with `--mlir-print-ir-after-all`
// and scope FileCheck to the ConvertSdeToArts dump window via `awk` so we
// inspect the exact post-lowering IR, uncluttered by later-cleaned
// variants (arts-dce may prune EDTs whose bodies fold to nothing, which is
// orthogonal to this lowering's correctness). VerifySdeLowered runs right
// after ConvertSdeToArts and rejects any residual `sde.*` op, so a
// successful end-to-end run also implies the lowering was complete.
//
// RUN: %carts-compile %s --arts-config %arts_config \
// RUN:   --pipeline openmp-to-arts --start-from openmp-to-arts \
// RUN:   --mlir-print-ir-after-all 2>&1 \
// RUN:   | awk '/IR Dump After ConvertSdeToArts/,/IR Dump After VerifySdeLowered/' \
// RUN:   | %FileCheck %s

module {
  // --------------------------------------------------------------------------
  // Case 1 — write-only codelet. `deps.c` Task 1 pattern: produce a new
  // tensor. The tensor.insert inside the codelet forces a destination-
  // passing materialize on the acquired DB memref.
  // --------------------------------------------------------------------------
  // CHECK-LABEL: func.func @codelet_write_only
  // CHECK: arts.db_alloc
  // CHECK: arts.db_acquire[<out>]
  // CHECK: arts.edt <task> <intranode>
  // CHECK: memref.store
  func.func @codelet_write_only() -> tensor<8xi32> {
    %d = sde.mu_data shared : tensor<8xi32>

    %token = sde.mu_token <write> %d
      : tensor<8xi32> -> !sde.token<tensor<8xi32>>

    %r = sde.cu_codelet (%token : !sde.token<tensor<8xi32>>)
        -> (tensor<8xi32>) {
    ^bb0(%arg: tensor<8xi32>):
      %c0 = arith.constant 0 : index
      %c42 = arith.constant 42 : i32
      %updated = tensor.insert %c42 into %arg[%c0] : tensor<8xi32>
      sde.yield %updated : tensor<8xi32>
    }

    func.return %r : tensor<8xi32>
  }

  // --------------------------------------------------------------------------
  // Case 2 — read + write codelet. Task 2 of `deps.c`: consume one tensor,
  // produce another. Two db_allocs, two db_acquires with distinct modes.
  // --------------------------------------------------------------------------
  // CHECK-LABEL: func.func @codelet_read_write
  // CHECK: arts.db_alloc
  // CHECK: arts.db_alloc
  // CHECK-DAG: arts.db_acquire[<in>]
  // CHECK-DAG: arts.db_acquire[<out>]
  // CHECK: arts.edt <task> <intranode>
  // CHECK: memref.store
  func.func @codelet_read_write() -> tensor<8xi32> {
    %src = sde.mu_data shared : tensor<8xi32>
    %dst = sde.mu_data shared : tensor<8xi32>

    %tsrc = sde.mu_token <read> %src
      : tensor<8xi32> -> !sde.token<tensor<8xi32>>
    %tdst = sde.mu_token <write> %dst
      : tensor<8xi32> -> !sde.token<tensor<8xi32>>

    %r = sde.cu_codelet (%tsrc, %tdst
          : !sde.token<tensor<8xi32>>,
            !sde.token<tensor<8xi32>>) -> (tensor<8xi32>) {
    ^bb0(%ain: tensor<8xi32>, %aout: tensor<8xi32>):
      %c0 = arith.constant 0 : index
      %v = tensor.extract %ain[%c0] : tensor<8xi32>
      %updated = tensor.insert %v into %aout[%c0] : tensor<8xi32>
      sde.yield %updated : tensor<8xi32>
    }
    func.return %r : tensor<8xi32>
  }

  // --------------------------------------------------------------------------
  // Case 3 — read-only codelet. No yielded result. A <read> token lowers
  // to a single `arts.db_acquire[<in>]` feeding an `arts.edt` with no
  // materialize (RFC V7/V10: read-only tokens have no destination-passing
  // counterpart).
  // --------------------------------------------------------------------------
  // CHECK-LABEL: func.func @codelet_read_only
  // CHECK: arts.db_alloc
  // CHECK: arts.db_acquire[<in>]
  // CHECK: arts.edt <task> <intranode>
  // CHECK-NOT: sde.
  func.func @codelet_read_only() {
    %d = sde.mu_data shared : tensor<8xi32>

    %t = sde.mu_token <read> %d
      : tensor<8xi32> -> !sde.token<tensor<8xi32>>

    sde.cu_codelet (%t : !sde.token<tensor<8xi32>>) {
    ^bb0(%arg: tensor<8xi32>):
      // Observable use so the codelet body isn't DCE'd to empty.
      %zero = arith.constant 0 : index
      %v = tensor.extract %arg[%zero] : tensor<8xi32>
      %buf = memref.alloca() : memref<1xi32>
      memref.store %v, %buf[%zero] : memref<1xi32>
      sde.yield
    }
    func.return
  }

  // --------------------------------------------------------------------------
  // Case 4 — readwrite codelet. In-place update, destination-passing style.
  // The acquire carries `<inout>` and the codelet yields a rebuilt tensor
  // that `materialize_in_destination` plumbs back into the same DB handle.
  // --------------------------------------------------------------------------
  // CHECK-LABEL: func.func @codelet_readwrite
  // CHECK: arts.db_alloc
  // CHECK: arts.db_acquire[<inout>]
  // CHECK: arts.edt <task> <intranode>
  // CHECK: memref.store
  func.func @codelet_readwrite() -> tensor<8xi32> {
    %d = sde.mu_data shared : tensor<8xi32>

    %t = sde.mu_token <readwrite> %d
      : tensor<8xi32> -> !sde.token<tensor<8xi32>>

    %r = sde.cu_codelet (%t : !sde.token<tensor<8xi32>>)
        -> (tensor<8xi32>) {
    ^bb0(%arg: tensor<8xi32>):
      %c0 = arith.constant 0 : index
      %v = tensor.extract %arg[%c0] : tensor<8xi32>
      %doubled = arith.addi %v, %v : i32
      %updated = tensor.insert %doubled into %arg[%c0] : tensor<8xi32>
      sde.yield %updated : tensor<8xi32>
    }
    func.return %r : tensor<8xi32>
  }

  // --------------------------------------------------------------------------
  // Case 5 — slice token. Element-space offsets/sizes carry over to the
  // acquire via the `partitioning` block attribute.
  // --------------------------------------------------------------------------
  // CHECK-LABEL: func.func @codelet_slice_read
  // CHECK: arts.db_alloc
  // CHECK: arts.db_acquire[<in>]
  // CHECK-SAME: partitioning(<block>
  // CHECK: arts.edt <task> <intranode>
  func.func @codelet_slice_read() {
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index

    %d = sde.mu_data shared : tensor<1024xf32>

    %t = sde.mu_token <read> %d [%c0] size [%c512]
      : tensor<1024xf32> -> !sde.token<tensor<512xf32>>

    sde.cu_codelet (%t : !sde.token<tensor<512xf32>>) {
    ^bb0(%arg: tensor<512xf32>):
      // Observable use so the codelet body isn't DCE'd to empty.
      %zero = arith.constant 0 : index
      %v = tensor.extract %arg[%zero] : tensor<512xf32>
      %buf = memref.alloca() : memref<1xf32>
      memref.store %v, %buf[%zero] : memref<1xf32>
      sde.yield
    }
    func.return
  }
}
