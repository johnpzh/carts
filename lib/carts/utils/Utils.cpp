///==========================================================================///
/// File: Utils.cpp
/// Defines dialect-neutral CARTS IR utility functions.
///==========================================================================///
#include "carts/utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace carts {

///===----------------------------------------------------------------------===///
/// Constant Index Creation Utilities
///===----------------------------------------------------------------------===///

Value createConstantIndex(OpBuilder &builder, Location loc, int64_t val) {
  return arith::ConstantIndexOp::create(builder, loc, val);
}

Value createZeroIndex(OpBuilder &builder, Location loc) {
  return createConstantIndex(builder, loc, 0);
}

Value createOneIndex(OpBuilder &builder, Location loc) {
  return createConstantIndex(builder, loc, 1);
}

bool isSideEffectFreeArithmeticLikeOp(Operation *op) {
  if (!op || op->hasTrait<OpTrait::IsTerminator>())
    return true;

  return isa<arith::ConstantOp, arith::AddIOp, arith::SubIOp, arith::MulIOp,
             arith::DivUIOp, arith::DivSIOp, arith::RemUIOp, arith::RemSIOp,
             arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
             arith::NegFOp, arith::CmpIOp, arith::CmpFOp, arith::SelectOp,
             arith::MaxUIOp, arith::MinUIOp, arith::MaximumFOp,
             arith::MinimumFOp, arith::IndexCastOp, arith::IndexCastUIOp,
             arith::ExtSIOp, arith::ExtUIOp, arith::ExtFOp, arith::TruncIOp,
             arith::TruncFOp, arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp,
             arith::FPToUIOp>(op);
}
} // namespace carts
} // namespace mlir
