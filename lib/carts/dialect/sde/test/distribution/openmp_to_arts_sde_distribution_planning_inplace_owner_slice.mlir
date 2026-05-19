// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from sde-planning --pipeline codir-to-arts --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=SDE
// RUN: %carts-compile %s --O3 --arts-config %arts_config --start-from sde-planning --pipeline create-dbs --mlir-print-ir-after-all 2>&1 | %FileCheck %s --check-prefix=DB

// In-place row-local kernels read and write the same root memref, but every
// access stays within the owner row. SDE may author the owner-slice plan during
// distribution planning. Until SDE/CODIR materializes token-local reduction
// views, the unsupported physical storage attrs are demoted before the raw
// CreateDbs bridge. Owner-local pipeline slices use one hardware wave, so an
// 8-worker single-node config maps 128 rows into 16 rows per task.

// SDE-LABEL: // -----// IR Dump After DistributionPlanning (distribution-planning) //----- //
// SDE: func.func @main
// SDE: sde.su_distribute <blocked> {
// SDE: sde.su_iterate (%c0) to (%c128) step (%{{.+}}) classification(<elementwise_pipeline>) {
// SDE: memref.load %{{.+}}[%arg{{[0-9]+}}, %arg{{[0-9]+}}] : memref<128x64xf32>
// SDE: memref.store %{{.+}}, %{{.+}}[%arg{{[0-9]+}}, %arg{{[0-9]+}}] : memref<128x64xf32>
// SDE: } {
// SDE-SAME: iterationTopology = #sde.iteration_topology<owner_strip>
// SDE-SAME: pattern = #sde.pattern<elementwise_pipeline>
// SDE-SAME: physicalBlockShape = [16, 64]
// SDE-SAME: physicalOwnerDims = [0]
// SDE-LABEL: // -----// IR Dump After IterationSpaceDecomposition

// DB-LABEL: // -----// IR Dump After CreateDbs
// DB: func.func @main
// DB: depPattern = #arts.dep_pattern<elementwise_pipeline>
// DB-SAME: planOwnerDims = [0]
// DB-SAME: planPhysicalBlockShape = [16, 64]
// DB-NOT: planHaloShape
// DB-NOT: SDE-authored physical DB layout reached CreateDbs as a raw memref

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @main() -> i32 {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %zero = arith.constant 0.000000e+00 : f32
    %scale = arith.constant 1.562500e-02 : f32
    %A = memref.alloc() : memref<128x64xf32>
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%row) : index = (%c0) to (%c128) step (%c1) {
          %sum = memref.alloca() : memref<f32>
          memref.store %zero, %sum[] : memref<f32>
          scf.for %col = %c0 to %c64 step %c1 {
            %v = memref.load %A[%row, %col] : memref<128x64xf32>
            %old = memref.load %sum[] : memref<f32>
            %next = arith.addf %old, %v : f32
            memref.store %next, %sum[] : memref<f32>
          }
          %total = memref.load %sum[] : memref<f32>
          %mean = arith.mulf %total, %scale : f32
          scf.for %col = %c0 to %c64 step %c1 {
            %v = memref.load %A[%row, %col] : memref<128x64xf32>
            %centered = arith.subf %v, %mean : f32
            memref.store %centered, %A[%row, %col] : memref<128x64xf32>
          }
          omp.yield
        }
      }
      omp.terminator
    }
    %check = memref.load %A[%c0, %c0] : memref<128x64xf32>
    func.call @use(%check) : (f32) -> ()
    memref.dealloc %A : memref<128x64xf32>
    return %c0_i32 : i32
  }

  func.func private @use(f32) attributes {llvm.linkage = #llvm.linkage<external>}
}
