///==========================================================================///
/// File: Passes.h
///
/// Pass declarations for the CODIR (Codelet IR) dialect.
///==========================================================================///

#ifndef CARTS_DIALECT_CODIR_TRANSFORMS_PASSES_H
#define CARTS_DIALECT_CODIR_TRANSFORMS_PASSES_H

#include "arts/Dialect.h"
#include "arts/dialect/sde/IR/SdeDialect.h"
#include "carts/dialect/codir/IR/CodirDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

namespace mlir::carts::codir {

std::unique_ptr<Pass> createConvertSdeToCodirPass();
std::unique_ptr<Pass> createVerifyCodirPass();
std::unique_ptr<Pass> createConvertCodirToArtsPass();

#define GEN_PASS_DECL
#include "carts/dialect/codir/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "carts/dialect/codir/Transforms/Passes.h.inc"

} // namespace mlir::carts::codir

#endif // CARTS_DIALECT_CODIR_TRANSFORMS_PASSES_H
