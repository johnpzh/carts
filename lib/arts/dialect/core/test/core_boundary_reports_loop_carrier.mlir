// RUN: %carts-compile %s --arts-config %arts_config \
// RUN:   --pipeline openmp-to-arts --start-from openmp-to-arts \
// RUN:   --report-core-object-boundary 2>&1 | %FileCheck %s
// RUN: %carts-compile %s --arts-config %arts_config \
// RUN:   --pipeline openmp-to-arts --start-from openmp-to-arts 2>&1 \
// RUN:   | %FileCheck %s --check-prefix=NO-REPORT

// CHECK-DAG: core-boundary-report: Core boundary still contains semantic parallel arts.edt wrapper
// CHECK-DAG: core-boundary-report: Core boundary still contains arts.for loop carrier
// CHECK-DAG: inside parent arts.edt type #arts.edt_type<parallel>
// CHECK-DAG: current compatibility producer is SDE-to-ARTS su_iterate lowering

// NO-REPORT-NOT: core-boundary-report:
// NO-REPORT: module
// NO-REPORT-NOT: core-boundary-report:

module {
  func.func @boundary_debt() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %buffer = memref.alloc() : memref<8xf32>
    %value = arith.constant 1.000000e+00 : f32

    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%i) : index = (%c0) to (%c8) step (%c1) {
          memref.store %value, %buffer[%i] : memref<8xf32>
          omp.yield
        }
      }
      omp.terminator
    }

    memref.dealloc %buffer : memref<8xf32>
    return
  }
}
