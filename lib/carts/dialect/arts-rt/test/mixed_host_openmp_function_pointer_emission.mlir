// RUN: %carts-compile %s --arts-config %arts_config --start-from arts-rt-to-llvm --emit-llvm -o - \
// RUN:   | %FileCheck %s

// Mixed host-OpenMP fallback modules still need ARTS EDT launches for work that
// was not selected for fallback. LLVM emission must materialize ARTS function
// pointers before ConvertOpenMPToLLVM rewrites func.func symbols to llvm.func,
// or polygeist.get_func verification fails before ConvertPolygeistToLLVM runs.

// CHECK: define{{.*}} @__arts_edt_1
// CHECK: define{{.*}} @main
// CHECK: @__arts_edt_1

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func private @__arts_edt_1(%argc: i32, %argv: !llvm.ptr, %depc: i32, %depv: !llvm.ptr) {
    return
  }

  func.func private @use_ptr(%fn: !llvm.ptr) {
    return
  }

  func.func @main() {
    %fn = polygeist.get_func @__arts_edt_1 : !llvm.ptr
    func.call @use_ptr(%fn) : (!llvm.ptr) -> ()
    omp.parallel {
      omp.terminator
    }
    return
  }
}
