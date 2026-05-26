// RUN: %carts-compile %s --arts-config %arts_config --start-from arts-rt-to-llvm --pipeline arts-rt-to-llvm | %FileCheck %s

// A large uniform DB split across many runtime DBs may have sub-16MiB
// per-partition payloads. ARTS-RT-to-LLVM should still choose interleaved
// placement when the aggregate allocation is large enough.

// CHECK-LABEL: func.func @uniform_multi_db_aggregate_interleaves
// CHECK: arts_db_create_with_guid_interleaved

module attributes {arts.runtime_total_nodes = 1 : i64, arts.runtime_total_workers = 64 : i64, dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @uniform_multi_db_aggregate_interleaves() {
    %route = arith.constant -1 : i32
    %c64 = arith.constant 64 : index
    %c8 = arith.constant 8 : index
    %c512 = arith.constant 512 : index
    %c784 = arith.constant 784 : index
    %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>, <block>, <uniform>] route(%route : i32) sizes[%c64] elementType(f32) elementSizes[%c8, %c512, %c784] {depPattern = #arts.dep_pattern<uniform>, distribution_kind = #arts.distribution_kind<block>, distribution_pattern = #arts.distribution_pattern<uniform>, distribution_version = 1 : i32, planIterationTopology = #arts.plan_iteration_topology<owner_strip>, planLogicalWorkerSlice = [8, 512, 784], planOwnerDims = [0], planPhysicalBlockShape = [8, 512, 784]} : (memref<?xi64>, memref<?xmemref<?x?x?xf32>>)
    return
  }
}
