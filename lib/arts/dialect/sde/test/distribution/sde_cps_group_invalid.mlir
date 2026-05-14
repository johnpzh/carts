// RUN: not %carts-compile %s --arts-config %arts_config --start-from openmp-to-arts --pipeline openmp-to-arts 2>&1 | %FileCheck %s

// CHECK: sde.cps group 0 has 1 stage(s), expected 2
// CHECK: sde.cps group 0 is missing sde.cps_stage_index 1

module {
  func.func @missing_cps_stage(%A: memref<16xf64>, %B: memref<16xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    arts_sde.cu_region <parallel> scope(<local>) {
      arts_sde.su_iterate (%c0) to (%c16) step (%c1) {
      ^bb0(%i: index):
        %v = memref.load %A[%i] : memref<16xf64>
        memref.store %v, %B[%i] : memref<16xf64>
        arts_sde.yield
      } {asyncStrategy = #arts_sde.async_strategy<cps_chain>,
         cps_group_id = 0 : i64,
         cps_stage_count = 2 : i64,
         cps_stage_index = 0 : i64,
         repetitionStructure = #arts_sde.repetition_structure<full_timestep>}
      arts_sde.yield
    }
    return
  }
}
