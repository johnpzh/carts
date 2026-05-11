// XFAIL: *
// FIXME: requires tests/inputs/snapshots/activations_openmp_to_arts.mlir which
// was never committed. Re-enable once the snapshot fixture is regenerated.
// RUN: %carts-compile %inputs_dir/snapshots/activations_openmp_to_arts.mlir --pipeline post-db-refinement --arts-config %inputs_dir/arts_1t.cfg | %FileCheck %s

// Verify that the late DbModeTightening pass does not degrade write-capable
// activation outputs back to read-only acquires after post-db-refinement rewrites
// the accesses through arts.db_ref. Under the single-worker contract these
// outputs should stay coarse, and tighten to `<out>` or `<inout>` (never `<in>`).
//
// Activations with tensor expansion (Tiling Phase 6F) that produce
// owner_dims annotations get `<out>`. Others stay `<inout>` because
// DbModeTightening conservatively keeps inout when no ownership proof exists.
// No activation output should ever be `<in>`.

// CHECK: arts.db_alloc[<inout>, <heap>, <write>, <coarse>
// CHECK: arts.db_alloc[<inout>, <heap>, <write>, <coarse>
// CHECK: arts.db_alloc[<inout>, <heap>, <write>, <coarse>
// CHECK: arts.db_alloc[<inout>, <heap>, <write>, <coarse>

// At least one activation output tightens to <out> (pure write with owner proof).
// CHECK: arts.db_acquire[<out>]
// The remaining acquires are <out> or <inout> — never <in>.
// CHECK-NOT: arts.db_acquire[<in>]
