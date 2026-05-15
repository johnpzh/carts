// RUN: not %carts-compile %s --arts-config %arts_config --start-from sde-planning --pipeline codir-to-arts 2>&1 | %FileCheck %s

// CHECK-COUNT-2: sde.cps candidate plan requires sde.cps_candidate_group_id, sde.cps_candidate_stage_index, sde.cps_candidate_stage_count, and sde.cps_candidate_requires_tokenized_dataflow together
// CHECK: sde.cps_candidate_requires_tokenized_dataflow must be a unit attr
// CHECK-DAG: sde.cps candidate timestep barrier requires sde.control_token produced after the previous candidate stage
// CHECK-DAG: sde.cps candidate stage pair requires sde.control_token boundary before successor stage

module {
  func.func @missing_cps_candidate_attrs(%A: memref<16xf64>, %B: memref<16xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    sde.su_iterate (%c0) to (%c16) step (%c1) {
    ^bb0(%i: index):
      %v = memref.load %A[%i] : memref<16xf64>
      memref.store %v, %B[%i] : memref<16xf64>
      sde.yield
    } {cps_candidate_group_id = 0 : i64}
    return
  }

  func.func @partial_cps_candidate_attrs_are_not_overwritten(%A: memref<16xf64>, %B: memref<16xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    sde.cu_region <parallel> scope(<local>) {
      sde.su_iterate (%c0) to (%c16) step (%c1) {
      ^bb0(%i: index):
        %v = memref.load %A[%i] : memref<16xf64>
        memref.store %v, %B[%i] : memref<16xf64>
        sde.yield
      } {cps_candidate_stage_index = 7 : i64}
      sde.su_barrier
      sde.su_iterate (%c0) to (%c16) step (%c1) {
      ^bb0(%i: index):
        %v = memref.load %B[%i] : memref<16xf64>
        memref.store %v, %A[%i] : memref<16xf64>
        sde.yield
      }
      sde.yield
    }
    return
  }

  func.func @candidate_requires_marker_is_unit(%A: memref<16xf64>, %B: memref<16xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    sde.su_iterate (%c0) to (%c16) step (%c1) {
    ^bb0(%i: index):
      %v = memref.load %A[%i] : memref<16xf64>
      memref.store %v, %B[%i] : memref<16xf64>
      sde.yield
    } {asyncStrategy = #sde.async_strategy<advance_edt>,
       cps_candidate_group_id = 0 : i64,
       cps_candidate_requires_tokenized_dataflow = 1 : i64,
       cps_candidate_stage_count = 1 : i64,
       cps_candidate_stage_index = 0 : i64,
       repetitionStructure = #sde.repetition_structure<full_timestep>}
    return
  }

  func.func @candidate_timestep_barrier_requires_control_token(%A: memref<16xf64>, %B: memref<16xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    sde.cu_region <parallel> scope(<local>) {
      sde.su_iterate (%c0) to (%c16) step (%c1) {
      ^bb0(%i: index):
        %v = memref.load %A[%i] : memref<16xf64>
        memref.store %v, %B[%i] : memref<16xf64>
        sde.yield
      } {asyncStrategy = #sde.async_strategy<advance_edt>,
         cps_candidate_group_id = 42 : i64,
         cps_candidate_requires_tokenized_dataflow,
         cps_candidate_stage_count = 2 : i64,
         cps_candidate_stage_index = 0 : i64,
         pattern = #sde.pattern<uniform>,
         physicalBlockShape = [16],
         physicalOwnerDims = [0],
         repetitionStructure = #sde.repetition_structure<full_timestep>,
         structuredClassification = #sde.structured_classification<elementwise>}
      sde.su_barrier
      sde.su_iterate (%c0) to (%c16) step (%c1) {
      ^bb0(%i: index):
        %v = memref.load %B[%i] : memref<16xf64>
        memref.store %v, %A[%i] : memref<16xf64>
        sde.yield
      } {asyncStrategy = #sde.async_strategy<advance_edt>,
         cps_candidate_group_id = 42 : i64,
         cps_candidate_requires_tokenized_dataflow,
         cps_candidate_stage_count = 2 : i64,
         cps_candidate_stage_index = 1 : i64,
         pattern = #sde.pattern<uniform>,
         physicalBlockShape = [16],
         physicalOwnerDims = [0],
         repetitionStructure = #sde.repetition_structure<full_timestep>,
         structuredClassification = #sde.structured_classification<elementwise>}
      sde.yield
    }
    return
  }

  func.func @candidate_adjacent_pair_requires_control_boundary(%A: memref<16xf64>, %B: memref<16xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c15 = arith.constant 15 : index
    %c16 = arith.constant 16 : index
    sde.su_iterate (%c0) to (%c15) step (%c1) {
    ^bb0(%i: index):
      %v = memref.load %A[%i] : memref<16xf64>
      memref.store %v, %B[%i] : memref<16xf64>
      sde.yield
    } {asyncStrategy = #sde.async_strategy<advance_edt>,
       cps_candidate_group_id = 43 : i64,
       cps_candidate_requires_tokenized_dataflow,
       cps_candidate_stage_count = 2 : i64,
       cps_candidate_stage_index = 0 : i64,
       repetitionStructure = #sde.repetition_structure<full_timestep>}
    sde.su_iterate (%c0) to (%c16) step (%c1) {
    ^bb0(%i: index):
      %v = memref.load %B[%i] : memref<16xf64>
      memref.store %v, %A[%i] : memref<16xf64>
      sde.yield
    } {asyncStrategy = #sde.async_strategy<advance_edt>,
       cps_candidate_group_id = 43 : i64,
       cps_candidate_requires_tokenized_dataflow,
       cps_candidate_stage_count = 2 : i64,
       cps_candidate_stage_index = 1 : i64,
       repetitionStructure = #sde.repetition_structure<full_timestep>}
    return
  }
}
