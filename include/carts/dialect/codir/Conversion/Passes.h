///==========================================================================///
/// File: Passes.h
///
/// Pass declarations for the SDE -> CODIR -> ARTS conversion boundaries.
///==========================================================================///

#ifndef CARTS_DIALECT_CODIR_CONVERSION_PASSES_H
#define CARTS_DIALECT_CODIR_CONVERSION_PASSES_H

#include "carts/Dialect.h"
#include "carts/dialect/sde/IR/SdeDialect.h"
#include "carts/dialect/codir/IR/CodirDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::carts::codir {

std::unique_ptr<Pass> createConvertSdeToCodirPass();
std::unique_ptr<Pass> createConvertCodirToArtsPass();

#define GEN_PASS_DECL
#include "carts/dialect/codir/Conversion/Passes.h.inc"

} // namespace mlir::carts::codir

#endif // CARTS_DIALECT_CODIR_CONVERSION_PASSES_H
