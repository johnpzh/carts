// RUN: %carts-compile %s --arts-config %arts_config --start-from late-concurrency-cleanup --pipeline late-concurrency-cleanup | %FileCheck %s

// Worker-local blocked slices materialize a non-negative lower bound as
// select(0 - blockBase < 0, 0, 0 - blockBase).  Both arms are aligned to the
// physical block span, so strip-mining should still recover a local inner loop
// and hoist db_ref out of the hot element loop.

// CHECK-LABEL: func.func @strip_mines_select_aligned_lower_bound
// CHECK: scf.for
// CHECK: %[[REF:.+]] = arts.db_ref
// CHECK: scf.for %[[LOCAL:arg[0-9]+]]
// CHECK-NOT: arith.divui
// CHECK-NOT: arith.remui
// CHECK: memref.load %[[REF]][%[[LOCAL]]]

module {
  func.func @strip_mines_select_aligned_lower_bound(%block: index,
                                                    %total: index) -> f64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c4 = arith.constant 4 : index
    %c-1_i32 = arith.constant -1 : i32
    %zero = arith.constant 0.000000e+00 : f64
    %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>, <block>, <uniform>] route(%c-1_i32 : i32) sizes[%c4] elementType(f64) elementSizes[%c64] : (memref<?xi64>, memref<?xmemref<?xf64>>)
    %base = arith.muli %block, %c64 : index
    %neg_base = arith.subi %c0, %base : index
    %below_zero = arith.cmpi slt, %neg_base, %c0 : index
    %lb = arith.select %below_zero, %c0, %neg_base : index
    %remaining = arith.subi %total, %base : index
    %past_end = arith.cmpi slt, %remaining, %c0 : index
    %remaining_nonnegative = arith.select %past_end, %c0, %remaining : index
    %ub = arith.minui %remaining_nonnegative, %c64 : index
    %sum = scf.for %i = %lb to %ub step %c1 iter_args(%acc = %zero) -> (f64) {
      %global = arith.addi %base, %i : index
      %block_idx = arith.divui %global, %c64 : index
      %local = arith.remui %global, %c64 : index
      %view = arts.db_ref %ptr[%block_idx] : memref<?xmemref<?xf64>> -> memref<?xf64>
      %value = memref.load %view[%local] : memref<?xf64>
      %next = arith.addf %acc, %value : f64
      scf.yield %next : f64
    }
    arts.db_free(%guid) : memref<?xi64>
    arts.db_free(%ptr) : memref<?xmemref<?xf64>>
    return %sum : f64
  }
}
