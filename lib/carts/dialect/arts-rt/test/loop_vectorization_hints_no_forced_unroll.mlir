// RUN: %carts-compile %s --arts-config %arts_config --start-from arts-rt-to-llvm --emit-llvm | %FileCheck %s

// ARTS-RT owns target-facing loop metadata. It may request vectorization on
// codelet-local data loops, but must not force unrolling: forced unroll
// metadata can make LLVM expand nested EDT loops into huge functions during the
// final benchmark build.

// CHECK-LABEL: define void @__arts_edt_vector_hint_no_forced_unroll
// CHECK: br label %{{.*}}, !llvm.loop ![[LOOP:[0-9]+]]
// CHECK: ![[LOOP]] = distinct !{![[LOOP]], ![[MUST:[0-9]+]], ![[VEC_ENABLE:[0-9]+]], ![[VEC_WIDTH:[0-9]+]], ![[INTERLEAVE:[0-9]+]]}
// CHECK: ![[MUST]] = !{!"llvm.loop.mustprogress"}
// CHECK: ![[VEC_ENABLE]] = !{!"llvm.loop.vectorize.enable", i1 true}
// CHECK: ![[VEC_WIDTH]] = !{!"llvm.loop.vectorize.width", i32 4}
// CHECK: ![[INTERLEAVE]] = !{!"llvm.loop.interleave.count", i32 4}
// CHECK-NOT: llvm.loop.unroll

module attributes {arts.runtime_total_nodes = 1 : i64, arts.runtime_total_workers = 8 : i64, dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<[8, 32]> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i16, dense<[16, 32]> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func private @__arts_edt_vector_hint_no_forced_unroll(%arg0: i32, %arg1: !llvm.ptr, %arg2: i32, %arg3: !llvm.ptr, %srcp: !llvm.ptr, %dstp: !llvm.ptr) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %src = polygeist.pointer2memref %srcp : !llvm.ptr to memref<?xf32>
    %dst = polygeist.pointer2memref %dstp : !llvm.ptr to memref<?xf32>
    scf.for %i = %c0 to %c64 step %c1 {
      %x = polygeist.load %src[%i] sizes(%c64) : memref<?xf32> -> f32
      %y = arith.mulf %x, %x : f32
      polygeist.store %y, %dst[%i] sizes(%c64) : f32, memref<?xf32>
    }
    return
  }
}
