// RUN: %carts-compile %s --arts-config %arts_config --start-from arts-rt-to-llvm --pipeline arts-rt-to-llvm | %FileCheck %s

// Lowering can leave db_ref operations in an already outlined function after
// arts_rt.dep_db_acquire has been lowered to raw depv pointer-table storage.
// ARTS-RT-to-LLVM must consume that residual db_ref before the source db_acquire
// is rewritten away.

// CHECK-LABEL: func.func @residual_db_ref_from_depv
// CHECK-NOT: arts.db_ref
// CHECK: %[[SLOT:.+]] = llvm.getelementptr
// CHECK: %[[PAYLOAD:.+]] = llvm.load %[[SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK: %[[VALUE:.+]] = llvm.load %[[PAYLOAD]] : !llvm.ptr -> f64
// CHECK: return %[[VALUE]] : f64

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<[8, 32]> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i16, dense<[16, 32]> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @residual_db_ref_from_depv(%depv: !llvm.ptr) -> f64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %guid, %ptr = arts_rt.dep_db_acquire(%depv) offset[%c0 : index] : !llvm.ptr -> memref<?xi64>, memref<?xmemref<?xf64>>
    %scratch = memref.alloca() : memref<1xi64>
    %scratch_cast = memref.cast %scratch : memref<1xi64> to memref<?xi64>
    %acq_guid, %acq_ptr = arts.db_acquire[<in>] (%scratch_cast : memref<?xi64>, %ptr : memref<?xmemref<?xf64>>) partitioning(<coarse>), indices[], offsets[%c0], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?xf64>>)
    %view = arts.db_ref %acq_ptr[%c0] : memref<?xmemref<?xf64>> -> memref<?xf64>
    %value = memref.load %view[%c0] : memref<?xf64>
    arts.db_release(%acq_ptr) : memref<?xmemref<?xf64>>
    return %value : f64
  }
}
