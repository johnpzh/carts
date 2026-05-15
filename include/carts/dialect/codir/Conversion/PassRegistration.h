///==========================================================================///
/// File: PassRegistration.h
///
/// TableGen-backed registration for SDE -> CODIR -> ARTS boundary passes.
///==========================================================================///

#ifndef CARTS_DIALECT_CODIR_CONVERSION_PASSREGISTRATION_H
#define CARTS_DIALECT_CODIR_CONVERSION_PASSREGISTRATION_H

#include "carts/dialect/codir/Conversion/Passes.h"

#include "mlir/Pass/PassRegistry.h"

namespace mlir::carts::codir {

#define GEN_PASS_REGISTRATION
#include "carts/dialect/codir/Conversion/Passes.h.inc"

} // namespace mlir::carts::codir

#endif // CARTS_DIALECT_CODIR_CONVERSION_PASSREGISTRATION_H
