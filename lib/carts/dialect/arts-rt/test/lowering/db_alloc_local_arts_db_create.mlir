// RUN: %carts-compile %s --arts-config %arts_config --start-from arts-rt-to-llvm --pipeline arts-rt-to-llvm | %FileCheck %s

// Locally owned DataBlocks are created with arts_db_create: the runtime mints
// the GUID at creation and hands the payload pointer back through the
// out-param, so the lowering never reserves a GUID or calls
// arts_db_create_with_guid. The alloc's route flows into hint->route.
// Distributed allocations keep the reserve-then-create protocol because every
// node must agree on the partition GUIDs before the owner creates them.

// CHECK-LABEL: func.func @local_db
// CHECK-NOT: call @arts_guid_reserve
// CHECK-DAG: %[[ROUTE:.*]] = arith.constant -1 : i32
// CHECK-DAG: %[[TYPE:.*]] = arith.constant 0 : i32
// CHECK: llvm.insertvalue %[[ROUTE]], {{.*}}[0] : !llvm.struct<(i32, i64)>
// CHECK: llvm.store {{.*}}, %[[HINT:.*]] : !llvm.struct<(i32, i64)>, !llvm.ptr
// CHECK: call @arts_db_create(%{{.*}}, %{{.*}}, %[[TYPE]], %[[HINT]])
// CHECK-NOT: call @arts_db_create_with_guid

// An explicit route is passed through as hint->route: arts_db_create creates
// on that rank (remotely when it is not the current node).
// CHECK-LABEL: func.func @routed_db
// CHECK-NOT: call @arts_guid_reserve
// CHECK-DAG: %[[ROUTE3:.*]] = arith.constant 3 : i32
// CHECK: llvm.insertvalue %[[ROUTE3]], {{.*}}[0] : !llvm.struct<(i32, i64)>
// CHECK: llvm.store {{.*}}, %[[HINT3:.*]] : !llvm.struct<(i32, i64)>, !llvm.ptr
// CHECK: call @arts_db_create(%{{.*}}, %{{.*}}, %{{.*}}, %[[HINT3]])
// CHECK-NOT: call @arts_db_create_with_guid

// Multi-element allocations create each DB inside the linearized loop.
// CHECK-LABEL: func.func @local_db_array
// CHECK-NOT: call @arts_guid_reserve
// CHECK: scf.for
// CHECK: call @arts_db_create(
// CHECK-NOT: call @arts_db_create_with_guid

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @local_db() -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %route = arith.constant -1 : i32
    %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>] route(%route : i32) sizes[%c1] elementType(f64) elementSizes[%c16] : (memref<?xi64>, memref<?x!llvm.ptr>)
    %guid_value = memref.load %guid[%c0] : memref<?xi64>
    return %guid_value : i64
  }
  func.func @routed_db() -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %route = arith.constant 3 : i32
    %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>] route(%route : i32) sizes[%c1] elementType(f64) elementSizes[%c16] : (memref<?xi64>, memref<?x!llvm.ptr>)
    %guid_value = memref.load %guid[%c0] : memref<?xi64>
    return %guid_value : i64
  }
  func.func @local_db_array() -> i64 {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %route = arith.constant -1 : i32
    %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>] route(%route : i32) sizes[%c4] elementType(f64) elementSizes[%c16] : (memref<?xi64>, memref<?x!llvm.ptr>)
    %guid_value = memref.load %guid[%c0] : memref<?xi64>
    return %guid_value : i64
  }
}
