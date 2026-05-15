///==========================================================================///
/// File: PassRegistration.h
///
/// TableGen-backed pass registration for CODIR and its boundary conversions.
///==========================================================================///

#ifndef CARTS_DIALECT_CODIR_TRANSFORMS_PASSREGISTRATION_H
#define CARTS_DIALECT_CODIR_TRANSFORMS_PASSREGISTRATION_H

#include "carts/dialect/codir/Transforms/Passes.h"

#include "mlir/Pass/PassRegistry.h"

namespace mlir::carts::codir {

#define GEN_PASS_REGISTRATION
#include "carts/dialect/codir/Transforms/Passes.h.inc"

} // namespace mlir::carts::codir

#endif // CARTS_DIALECT_CODIR_TRANSFORMS_PASSREGISTRATION_H
