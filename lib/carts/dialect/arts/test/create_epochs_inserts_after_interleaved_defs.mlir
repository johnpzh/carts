// RUN: %carts-compile %s --pipeline epochs --start-from epochs --arts-config %inputs_dir/arts_multinode.cfg | %FileCheck %s

// A barrier segment may contain an early EDT launch, then a non-moved DB
// allocation, then a later EDT launch that depends on that DB. CreateEpochs
// must not insert the epoch before the interleaved allocation, otherwise moving
// the later EDT into the epoch breaks SSA dominance.

module {
  func.func @epoch_insertion_respects_interleaved_db_alloc_defs() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %route = arith.constant 0 : i32

    %guid0, %ptr0 = arts.db_alloc[<inout>, <heap>, <write>, <block>] route(%route : i32) sizes[%c4] elementType(f64) elementSizes[%c1] {planOwnerDims = [0], planPhysicalBlockShape = [1]} : (memref<?xi64>, memref<?xmemref<?xf64>>)
    scf.for %i = %c0 to %c4 step %c1 {
      %acq_guid0, %acq_ptr0 = arts.db_acquire[<inout>] (%guid0 : memref<?xi64>, %ptr0 : memref<?xmemref<?xf64>>) partitioning(<block>), indices[], offsets[%i], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?xf64>>)
      arts.edt <task> <intranode> route(%route) (%acq_ptr0) : memref<?xmemref<?xf64>> {
      ^bb0(%dep0: memref<?xmemref<?xf64>>):
        %inner_c0 = arith.constant 0 : index
        %value = arith.constant 1.000000e+00 : f64
        %payload = arts.db_ref %dep0[%inner_c0] : memref<?xmemref<?xf64>> -> memref<?xf64>
        memref.store %value, %payload[%inner_c0] : memref<?xf64>
        arts.yield
      }
    }

    %guid1, %ptr1 = arts.db_alloc[<inout>, <heap>, <write>, <block>] route(%route : i32) sizes[%c4] elementType(f64) elementSizes[%c1] {planOwnerDims = [0], planPhysicalBlockShape = [1]} : (memref<?xi64>, memref<?xmemref<?xf64>>)
    scf.for %i = %c0 to %c4 step %c1 {
      %acq_guid0, %acq_ptr0 = arts.db_acquire[<in>] (%guid0 : memref<?xi64>, %ptr0 : memref<?xmemref<?xf64>>) partitioning(<block>), indices[], offsets[%i], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?xf64>>)
      %acq_guid1, %acq_ptr1 = arts.db_acquire[<out>] (%guid1 : memref<?xi64>, %ptr1 : memref<?xmemref<?xf64>>) partitioning(<block>), indices[], offsets[%i], sizes[%c1] -> (memref<?xi64>, memref<?xmemref<?xf64>>)
      arts.edt <task> <internode> route(%route) (%acq_ptr0, %acq_ptr1) : memref<?xmemref<?xf64>>, memref<?xmemref<?xf64>> {
      ^bb0(%dep0: memref<?xmemref<?xf64>>, %dep1: memref<?xmemref<?xf64>>):
        %inner_c0 = arith.constant 0 : index
        %src = arts.db_ref %dep0[%inner_c0] : memref<?xmemref<?xf64>> -> memref<?xf64>
        %dst = arts.db_ref %dep1[%inner_c0] : memref<?xmemref<?xf64>> -> memref<?xf64>
        %value = memref.load %src[%inner_c0] : memref<?xf64>
        memref.store %value, %dst[%inner_c0] : memref<?xf64>
        arts.yield
      }
    }
    arts.barrier {barrierReason = #arts.barrier_reason<required_memory>}
    return
  }
}

// CHECK-LABEL: func.func @epoch_insertion_respects_interleaved_db_alloc_defs
// CHECK: arts.db_alloc
// CHECK: %[[GUID1:.*]], %[[PTR1:.*]] = arts.db_alloc
// CHECK-NEXT: %{{.*}} = arts.epoch
// CHECK: arts.db_acquire{{.*}}(%[[GUID1]] : {{.*}}, %[[PTR1]] : {{.*}})
