///==========================================================================///
/// File: Passes.td
///
/// Umbrella pass declarations for the SDE (Structured Decomposition
/// Environment) dialect. Individual pass groups live in phase-grouped
/// sub-files.
///==========================================================================///
#ifndef ARTS_DIALECT_SDE_TRANSFORMS_PASSES
#define ARTS_DIALECT_SDE_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

include "carts/dialect/sde/Transforms/StatePasses.td"
include "carts/dialect/sde/Transforms/DepPasses.td"
include "carts/dialect/sde/Transforms/EffectPasses.td"
include "carts/dialect/sde/Transforms/ConversionPasses.td"
include "carts/dialect/sde/Transforms/VerifyPasses.td"

#endif // ARTS_DIALECT_SDE_TRANSFORMS_PASSES
