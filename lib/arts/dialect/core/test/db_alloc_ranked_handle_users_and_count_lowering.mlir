// RUN: %carts-compile %s --arts-config %arts_config --start-from arts-to-llvm --pipeline arts-to-llvm | %FileCheck %s

// Multi-dimensional DB allocation lowering linearizes ranked handle users.
// The helper must route those load/store rewrites through PatternRewriter so
// the greedy driver does not keep stale worklist entries before it reaches
// nearby db_num_elements users.

// CHECK-LABEL: func.func @ranked_db_alloc_handle_users_and_count
// CHECK-NOT: arts.db_alloc
// CHECK-NOT: arts.db_num_elements
// CHECK: arith.constant 64 : index

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @ranked_db_alloc_handle_users_and_count() -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %route = arith.constant -1 : i32
    %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>, <block>] route(%route : i32) sizes[%c16, %c4] elementType(f64) elementSizes[%c1] : (memref<?x?xi64>, memref<?x?x!llvm.ptr>)
    %count = arts.db_num_elements sizes(%c16, %c4) : -> index
    %count_i32 = arith.index_cast %count : index to i32
    %guid_value = memref.load %guid[%c0, %c0] : memref<?x?xi64>
    %ptr_value = memref.load %ptr[%c0, %c0] : memref<?x?x!llvm.ptr>
    %ptr_int = llvm.ptrtoint %ptr_value : !llvm.ptr to i64
    %ptr_count = arith.index_cast %ptr_int : i64 to index
    %ptr_count_i32 = arith.index_cast %ptr_count : index to i32
    %guid_i32 = arith.trunci %guid_value : i64 to i32
    %sum0 = arith.addi %count_i32, %ptr_count_i32 : i32
    %sum1 = arith.addi %sum0, %guid_i32 : i32
    return %sum1 : i32
  }
}
