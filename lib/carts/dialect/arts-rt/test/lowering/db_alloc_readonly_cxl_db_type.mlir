// RUN: %carts-compile %s --arts-config %arts_config --cxl-readonly-dbs --start-from arts-rt-to-llvm --pipeline arts-rt-to-llvm | %FileCheck %s --check-prefix=CXL
// RUN: %carts-compile %s --arts-config %arts_config --start-from arts-rt-to-llvm --pipeline arts-rt-to-llvm | %FileCheck %s --check-prefix=DEFAULT

// Read-only DataBlocks (dbMode <read>) are created with the ARTS_DB_CXL
// runtime type (4) when --cxl-readonly-dbs is set. Write-mode DataBlocks and
// default compiles keep ARTS_DB_DEFAULT (0). The db_type is the third
// argument of arts_db_create_with_guid.

// CXL-LABEL: func.func @readonly_db
// CXL-DAG: %[[CXL_TYPE:.*]] = arith.constant 4 : i32
// CXL: call @arts_db_create_with_guid({{[^,]+}}, {{[^,]+}}, %[[CXL_TYPE]],

// CXL-LABEL: func.func @writable_db
// CXL-DAG: %[[DEFAULT_TYPE:.*]] = arith.constant 0 : i32
// CXL: call @arts_db_create_with_guid({{[^,]+}}, {{[^,]+}}, %[[DEFAULT_TYPE]],

// DEFAULT-LABEL: func.func @readonly_db
// DEFAULT-DAG: %[[DEFAULT_TYPE:.*]] = arith.constant 0 : i32
// DEFAULT: call @arts_db_create_with_guid({{[^,]+}}, {{[^,]+}}, %[[DEFAULT_TYPE]],

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @readonly_db() -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %route = arith.constant -1 : i32
    %guid, %ptr = arts.db_alloc[<in>, <heap>, <read>] route(%route : i32) sizes[%c1] elementType(f64) elementSizes[%c16] : (memref<?xi64>, memref<?x!llvm.ptr>)
    %guid_value = memref.load %guid[%c0] : memref<?xi64>
    return %guid_value : i64
  }
  func.func @writable_db() -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %route = arith.constant -1 : i32
    %guid, %ptr = arts.db_alloc[<inout>, <heap>, <write>] route(%route : i32) sizes[%c1] elementType(f64) elementSizes[%c16] : (memref<?xi64>, memref<?x!llvm.ptr>)
    %guid_value = memref.load %guid[%c0] : memref<?xi64>
    return %guid_value : i64
  }
}
