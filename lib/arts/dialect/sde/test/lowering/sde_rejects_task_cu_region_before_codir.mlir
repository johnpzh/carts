// RUN: not %carts-compile %s --O3 --arts-config %arts_config \
// RUN:   --start-from codir-to-arts --pipeline codir-to-arts 2>&1 \
// RUN:   | %FileCheck %s

// CODIR-to-ARTS rejects any unconverted SDE operation at the boundary.
// CHECK: SDE operation 'sde.cu_region' survived past boundary conversion

module {
  func.func @reject_task_cu_region() {
    sde.cu_region <task> {
      sde.yield
    }
    return
  }
}
