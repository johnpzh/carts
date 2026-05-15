// SDE-to-CODIR plus CODIR-to-ARTS must turn memref-native SDE MU/codelet ops
// into ARTS core:
//   - sde.mu_data     : memref<...> -> arts.db_alloc + arts.db_ref
//   - sde.mu_token    <mode>        -> codir dep view -> arts.db_acquire <mode>
//   - sde.cu_codelet  token body    -> codir.codelet -> arts.edt
//
// RUN: %carts-compile %s --arts-config %arts_config \
// RUN:   --pipeline codir-to-arts --start-from sde-planning \
// RUN:   --mlir-print-ir-after-all 2>&1 \
// RUN:   | awk '/IR Dump After ConvertCodirToArts/,/IR Dump After VerifySdeLowered/' \
// RUN:   | %FileCheck %s

module {
  // CHECK-LABEL: func.func @codelet_memref_readwrite
  // CHECK: arts.db_alloc
  // CHECK: arts.db_acquire[<inout>]
  // CHECK: arts.edt <task> <intranode>
  // CHECK: memref.store
  // CHECK-NOT: sde.
  func.func @codelet_memref_readwrite() {
    %d = sde.mu_data shared : memref<8xi32>

    %t = sde.mu_token <readwrite> %d
      : memref<8xi32> -> !sde.token<memref<8xi32>>

    sde.cu_codelet (%t : !sde.token<memref<8xi32>>) {
    ^bb0(%arg: memref<8xi32>):
      %c0 = arith.constant 0 : index
      %v = memref.load %arg[%c0] : memref<8xi32>
      %doubled = arith.addi %v, %v : i32
      memref.store %doubled, %arg[%c0] : memref<8xi32>
      sde.yield
    }
    func.return
  }

  // CHECK-LABEL: func.func @codelet_memref_slice_read
  // CHECK: arts.db_alloc
  // CHECK: arts.db_acquire[<in>]
  // CHECK-SAME: partitioning(<block>
  // CHECK: arts.edt <task> <intranode>
  // CHECK: memref.subview
  // CHECK: memref.load %{{.*}}[%{{.*}}] : memref<?xf32, strided<[1], offset: ?>>
  // CHECK-NOT: sde.
  func.func @codelet_memref_slice_read() {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c512 = arith.constant 512 : index
    %d = sde.mu_data shared : memref<1024xf32>

    %t = sde.mu_token <read> %d [%c16] size [%c512]
      : memref<1024xf32> -> !sde.token<memref<?xf32, strided<[1], offset: ?>>>

    sde.cu_codelet (%t : !sde.token<memref<?xf32, strided<[1], offset: ?>>>) {
    ^bb0(%arg: memref<?xf32, strided<[1], offset: ?>>):
      %zero = arith.constant 0 : index
      %v = memref.load %arg[%zero] : memref<?xf32, strided<[1], offset: ?>>
      %buf = memref.alloca() : memref<1xf32>
      memref.store %v, %buf[%zero] : memref<1xf32>
      sde.yield
    }
    func.return
  }

  // CHECK-LABEL: func.func @codelet_memref_slice_global_index_read
  // CHECK: arts.db_alloc
  // CHECK: arts.db_acquire[<in>]
  // CHECK-SAME: partitioning(<block>
  // CHECK: arts.edt <task> <intranode>
  // CHECK: arith.subi
  // CHECK: memref.load %{{.*}}[%{{.*}}] : memref<?xf32, strided<[1], offset: ?>>
  // CHECK-NOT: sde.
  func.func @codelet_memref_slice_global_index_read(%global_i: index) {
    %c16 = arith.constant 16 : index
    %c512 = arith.constant 512 : index
    %d = sde.mu_data shared : memref<1024xf32>

    %t = sde.mu_token <read> %d [%c16] size [%c512]
      : memref<1024xf32> -> !sde.token<memref<?xf32, strided<[1], offset: ?>>>

    sde.cu_codelet (%t : !sde.token<memref<?xf32, strided<[1], offset: ?>>>) captures(%global_i : index) {
    ^bb0(%arg: memref<?xf32, strided<[1], offset: ?>>, %global_i_arg: index):
      %c0 = arith.constant 0 : index
      %v = memref.load %arg[%global_i_arg] : memref<?xf32, strided<[1], offset: ?>>
      %buf = memref.alloca() : memref<1xf32>
      memref.store %v, %buf[%c0] : memref<1xf32>
      sde.yield
    }
    func.return
  }

  // CHECK-LABEL: func.func @codelet_memref_read_write
  // CHECK: arts.db_alloc
  // CHECK: arts.db_alloc
  // CHECK-DAG: arts.db_acquire[<in>]
  // CHECK-DAG: arts.db_acquire[<out>]
  // CHECK: arts.edt <task> <intranode>
  // CHECK: memref.store
  // CHECK-NOT: sde.
  func.func @codelet_memref_read_write() {
    %src = sde.mu_data shared : memref<8xi32>
    %dst = sde.mu_data shared : memref<8xi32>

    %tsrc = sde.mu_token <read> %src
      : memref<8xi32> -> !sde.token<memref<8xi32>>
    %tdst = sde.mu_token <write> %dst
      : memref<8xi32> -> !sde.token<memref<8xi32>>

    sde.cu_codelet (%tsrc, %tdst
          : !sde.token<memref<8xi32>>,
            !sde.token<memref<8xi32>>) {
    ^bb0(%ain: memref<8xi32>, %aout: memref<8xi32>):
      %c0 = arith.constant 0 : index
      %v = memref.load %ain[%c0] : memref<8xi32>
      memref.store %v, %aout[%c0] : memref<8xi32>
      sde.yield
    }
    func.return
  }
}
