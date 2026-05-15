// RUN: not %carts-compile %s --arts-config %arts_config --start-from=sde-to-codir --pipeline=sde-to-codir 2>&1 | %FileCheck %s

// `sde.mu_dep` is a source dependency declaration. It must be consumed before
// CODIR becomes the codelet boundary; unresolved declarations must not leak to
// CODIR-to-ARTS.

// CHECK: error: 'sde.mu_dep' op must be consumed by convert-sde-to-codir before the CODIR boundary

module {
  func.func @unresolved_mu_dep(%arg0: memref<8xi32>) {
    %dep = sde.mu_dep <read> %arg0 : memref<8xi32> -> !sde.dep
    return
  }
}
