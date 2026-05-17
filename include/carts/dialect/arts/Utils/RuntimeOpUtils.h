///==========================================================================///
/// File: RuntimeOpUtils.h
///
/// ARTS runtime-object helper predicates and builders.
///==========================================================================///

#ifndef CARTS_DIALECT_ARTS_UTILS_RUNTIMEOPUTILS_H
#define CARTS_DIALECT_ARTS_UTILS_RUNTIMEOPUTILS_H

#include "carts/Dialect.h"
#include "carts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace carts::arts {

/// Create the ARTS runtime route sentinel for "run/create on the current node"
/// when no explicit destination rank is requested.
inline Value createCurrentNodeRoute(OpBuilder &builder, Location loc) {
  constexpr int64_t kCurrentNodeRoute = -1;
  return arith::ConstantIntOp::create(builder, loc, kCurrentNodeRoute, 32);
}

/// Return true when `val` is produced by an ARTS runtime-topology query.
inline bool isArtsRuntimeQuery(Value val) {
  if (!val)
    return false;

  val = ValueAnalysis::stripNumericCasts(val);
  Operation *defOp = val.getDefiningOp();
  return isa_and_nonnull<RuntimeQueryOp>(defOp);
}

/// Return true when `op` is an undef-like operation (llvm.mlir.undef,
/// polygeist.undef, or arts.undef).
inline bool isUndefLikeOp(Operation *op) {
  if (!op)
    return false;
  StringRef name = op->getName().getStringRef();
  return name == "llvm.mlir.undef" || name == "polygeist.undef" ||
         name == "arts.undef";
}

} // namespace carts::arts
} // namespace mlir

#endif // CARTS_DIALECT_ARTS_UTILS_RUNTIMEOPUTILS_H
