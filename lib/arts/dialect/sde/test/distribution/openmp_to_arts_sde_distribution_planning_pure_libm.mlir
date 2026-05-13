// RUN: %carts-compile %s --O3 --arts-config %arts_config --pipeline openmp-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=SDE
// RUN: %carts-compile %s --O3 --arts-config %arts_config --pipeline create-dbs | %FileCheck %s --check-prefix=DB

// SDE treats known scalar libm calls as pure elementwise compute so output-only
// loops still receive the SDE-authored block plan before Core creates DBs.

// SDE-LABEL: // -----// IR Dump After DistributionPlanning (distribution-planning) //----- //
// SDE: func.func @main
// SDE: arts_sde.su_distribute <blocked> {
// SDE: arts_sde.su_iterate (%c0) to (%c128) step (%c1) classification(<elementwise>) {
// SDE: func.call @tanhf
// SDE: } {
// SDE-SAME: iterationTopology = #arts_sde.iteration_topology<owner_strip>
// SDE-SAME: physicalBlockShape = [16]
// SDE-SAME: physicalOwnerDims = [0]
// SDE-LABEL: // -----// IR Dump After IterationSpaceDecomposition

// DB-LABEL: func.func @main
// DB: arts.db_alloc
// DB-SAME: <block>
// DB-SAME: elementSizes[%c16]
// DB-SAME: planPhysicalBlockShape = [16]
// DB-NOT: arts.db_alloc{{.*}}<coarse>{{.*}}elementSizes[%c128]

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main() -> i32 {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %scale = arith.constant 1.250000e-01 : f32
    %B = memref.alloc() : memref<128xf32>
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c0) to (%c128) step (%c1) {
          %si = arith.index_cast %i : index to i32
          %xf = arith.sitofp %si : i32 to f32
          %x = arith.mulf %xf, %scale : f32
          %y = func.call @tanhf(%x) : (f32) -> f32
          memref.store %y, %B[%i] : memref<128xf32>
          omp.yield
        }
      }
      omp.terminator
    }
    %check = memref.load %B[%c0] : memref<128xf32>
    func.call @use(%check) : (f32) -> ()
    memref.dealloc %B : memref<128xf32>
    return %c0_i32 : i32
  }

  func.func private @use(f32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @tanhf(f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>}
}
