// RUN: %carts-compile %s --arts-config %inputs_dir/arts_multinode.cfg --start-from arts-rt-to-llvm --pipeline arts-rt-to-llvm | %FileCheck %s --check-prefix=DYNAMIC
// RUN: %carts-compile %s --arts-config %inputs_dir/arts_multinode.cfg --runtime-static-workers --start-from arts-rt-to-llvm --pipeline arts-rt-to-llvm | %FileCheck %s --check-prefix=STATIC

// `arts.runtime_query <total_workers>` is ARTS-level logical capacity across
// the configured cluster. The ARTS runtime API reports workers per node, so
// ARTS-RT multiplies by total nodes when keeping the query dynamic.

// DYNAMIC-LABEL: func.func @total_workers_query
// DYNAMIC-DAG: %[[LOCAL:.*]] = {{func[.]call|call}} @arts_get_total_workers() : () -> i32
// DYNAMIC-DAG: %[[NODES:.*]] = arith.constant 2 : i32
// DYNAMIC: %[[TOTAL:.*]] = arith.muli %[[LOCAL]], %[[NODES]] : i32
// DYNAMIC: return %[[TOTAL]] : i32

// STATIC-LABEL: func.func @total_workers_query
// STATIC-NOT: func.call @arts_get_total_workers
// STATIC: %[[TOTAL:.*]] = arith.constant 16 : i32
// STATIC: return %[[TOTAL]] : i32

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @total_workers_query() -> i32 {
    %workers = arts.runtime_query <total_workers> -> i32
    return %workers : i32
  }
}
