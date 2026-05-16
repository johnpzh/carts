// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from sde-planning --pipeline codir-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s
// RUN: %carts-compile %s --O3 --arts-config %inputs_dir/arts_multinode_4x16.cfg --runtime-static-workers --start-from sde-planning --pipeline create-dbs --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=STATIC

// SDE may express symbolic grain arithmetic with target-neutral resource
// queries. CODIR-to-ARTS binds those queries to the ARTS runtime mechanism.

// CHECK-LABEL: // -----// IR Dump After ConvertCodirToArts (convert-codir-to-arts) //----- //
// CHECK: func.func @resource_query
// CHECK: %[[WORKERS_I32:.*]] = arts.runtime_query <total_workers> -> i32
// CHECK: %[[WORKERS:.*]] = arith.index_cast %[[WORKERS_I32]] : i32 to index
// CHECK: arith.addi %[[WORKERS]], %{{.*}} : index
// CHECK-NOT: sde.resource_query

// STATIC-LABEL: // -----// IR Dump After PolygeistCanonicalize (canonicalize-polygeist) //----- //
// STATIC: func.func @resource_query
// STATIC: %c65 = arith.constant 65 : index
// STATIC: memref.store %c65

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @resource_query(%out: memref<1xindex>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %workers = sde.resource_query <logical_workers>
    %next = arith.addi %workers, %c1 : index
    memref.store %next, %out[%c0] : memref<1xindex>
    return
  }
}
