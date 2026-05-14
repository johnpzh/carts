// RUN: not %carts-compile %s --arts-config %arts_config --start-from openmp-to-arts --pipeline openmp-to-arts 2>&1 | %FileCheck %s

// CHECK-COUNT-2: sde.cps candidate plan requires sde.cps_candidate_group_id, sde.cps_candidate_stage_index, sde.cps_candidate_stage_count, and sde.cps_candidate_requires_tokenized_dataflow together
// CHECK: sde.cps_candidate_requires_tokenized_dataflow must be a unit attr

module {
  func.func @missing_cps_candidate_attrs(%A: memref<16xf64>, %B: memref<16xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    arts_sde.su_iterate (%c0) to (%c16) step (%c1) {
    ^bb0(%i: index):
      %v = memref.load %A[%i] : memref<16xf64>
      memref.store %v, %B[%i] : memref<16xf64>
      arts_sde.yield
    } {cps_candidate_group_id = 0 : i64}
    return
  }

  func.func @partial_cps_candidate_attrs_are_not_overwritten(%A: memref<16xf64>, %B: memref<16xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    arts_sde.cu_region <parallel> scope(<local>) {
      arts_sde.su_iterate (%c0) to (%c16) step (%c1) {
      ^bb0(%i: index):
        %v = memref.load %A[%i] : memref<16xf64>
        memref.store %v, %B[%i] : memref<16xf64>
        arts_sde.yield
      } {cps_candidate_stage_index = 7 : i64}
      arts_sde.su_barrier
      arts_sde.su_iterate (%c0) to (%c16) step (%c1) {
      ^bb0(%i: index):
        %v = memref.load %B[%i] : memref<16xf64>
        memref.store %v, %A[%i] : memref<16xf64>
        arts_sde.yield
      }
      arts_sde.yield
    }
    return
  }

  func.func @candidate_requires_marker_is_unit(%A: memref<16xf64>, %B: memref<16xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    arts_sde.su_iterate (%c0) to (%c16) step (%c1) {
    ^bb0(%i: index):
      %v = memref.load %A[%i] : memref<16xf64>
      memref.store %v, %B[%i] : memref<16xf64>
      arts_sde.yield
    } {asyncStrategy = #arts_sde.async_strategy<advance_edt>,
       cps_candidate_group_id = 0 : i64,
       cps_candidate_requires_tokenized_dataflow = 1 : i64,
       cps_candidate_stage_count = 1 : i64,
       cps_candidate_stage_index = 0 : i64,
       repetitionStructure = #arts_sde.repetition_structure<full_timestep>}
    return
  }
}
