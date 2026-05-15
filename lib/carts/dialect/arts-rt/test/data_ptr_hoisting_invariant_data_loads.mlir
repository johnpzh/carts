// RUN: %carts-compile %s --arts-config %arts_config --start-from arts-to-llvm --pipeline arts-to-llvm | %FileCheck %s

// CHECK-LABEL: func.func private @__arts_edt_hoist_invariant_data_load
// CHECK: scf.for %[[I:.+]] =
// CHECK: scf.for %[[K:.+]] =
// CHECK: %[[AIK:.+]] = polygeist.load %{{.+}}[%[[I]], %[[K]]] sizes
// CHECK-NEXT: scf.for %[[JT:.+]] =
// CHECK: scf.for %[[J:.+]] =
// CHECK: arith.mulf %[[AIK]],

// CHECK-LABEL: func.func private @__arts_edt_keep_same_root_data_load
// CHECK: scf.for %[[NI:.+]] =
// CHECK: scf.for %[[NK:.+]] =
// CHECK: scf.for %[[NJT:.+]] =
// CHECK: scf.for %[[NJ:.+]] =
// CHECK: polygeist.load %{{.+}}[%[[NI]], %[[NK]]] sizes
// CHECK: polygeist.store

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<[8, 32]> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i16, dense<[16, 32]> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func private @__arts_edt_hoist_invariant_data_load(%ap: !llvm.ptr, %bp: !llvm.ptr, %cp: !llvm.ptr) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %a = polygeist.pointer2memref %ap : !llvm.ptr to memref<?x?xf32>
    %b = polygeist.pointer2memref %bp : !llvm.ptr to memref<?x?xf32>
    %c = polygeist.pointer2memref %cp : !llvm.ptr to memref<?x?xf32>
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %k = %c0 to %c8 step %c1 {
        scf.for %jt = %c0 to %c8 step %c2 {
          %jEnd = arith.addi %jt, %c2 : index
          scf.for %j = %jt to %jEnd step %c1 {
            %aik = polygeist.load %a[%i, %k] sizes(%c8, %c8) : memref<?x?xf32> -> f32
            %bkj = polygeist.load %b[%k, %j] sizes(%c8, %c8) : memref<?x?xf32> -> f32
            %old = polygeist.load %c[%i, %j] sizes(%c8, %c8) : memref<?x?xf32> -> f32
            %prod = arith.mulf %aik, %bkj : f32
            %new = arith.addf %old, %prod : f32
            polygeist.store %new, %c[%i, %j] sizes(%c8, %c8) : f32, memref<?x?xf32>
          }
        }
      }
    }
    return
  }

  func.func private @__arts_edt_keep_same_root_data_load(%ap: !llvm.ptr) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %a = polygeist.pointer2memref %ap : !llvm.ptr to memref<?x?xf32>
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %k = %c0 to %c8 step %c1 {
        scf.for %jt = %c0 to %c8 step %c2 {
          %jEnd = arith.addi %jt, %c2 : index
          scf.for %j = %jt to %jEnd step %c1 {
            %aik = polygeist.load %a[%i, %k] sizes(%c8, %c8) : memref<?x?xf32> -> f32
            polygeist.store %aik, %a[%i, %j] sizes(%c8, %c8) : f32, memref<?x?xf32>
          }
        }
      }
    }
    return
  }
}
