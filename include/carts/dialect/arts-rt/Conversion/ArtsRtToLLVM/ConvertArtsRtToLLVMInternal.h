///==========================================================================///
/// File: ConvertArtsRtToLLVMInternal.h
///
/// Local implementation contract for ConvertArtsRtToLLVM. This header is
/// intentionally private to the arts-rt-to-llvm implementation split and
/// should not be used as shared compiler infrastructure.
///==========================================================================///

#ifndef CARTS_DIALECT_ARTS_RT_CONVERSION_ARTSRTTOLLVM_INTERNAL_H
#define CARTS_DIALECT_ARTS_RT_CONVERSION_ARTSRTTOLLVM_INTERNAL_H

#include "carts/Dialect.h"
#include "carts/dialect/arts-rt/Conversion/ArtsRtToLLVM/CodegenSupport.h"
#include "mlir/Conversion/LLVMCommon/StructBuilder.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "polygeist/Ops.h"

namespace mlir::carts::arts_rt::convert_arts_rt_to_llvm {

using ::mlir::carts::arts_rt::ArtsCodegen;

///===----------------------------------------------------------------------===///
/// ArtsRtToLLVMPattern base class
///===----------------------------------------------------------------------===///

template <typename OpType>
class ArtsRtToLLVMPattern : public OpRewritePattern<OpType> {
protected:
  ArtsCodegen *AC;

public:
  ArtsRtToLLVMPattern(MLIRContext *context, ArtsCodegen *ac)
      : OpRewritePattern<OpType>(context), AC(ac) {}
};

///===----------------------------------------------------------------------===///
/// ArtsHintBuilder
///===----------------------------------------------------------------------===///

class ArtsHintBuilder : public StructBuilder {
public:
  explicit ArtsHintBuilder(Value value) : StructBuilder(value) {}

  static ArtsHintBuilder undef(OpBuilder &builder, Location loc,
                               Type descriptorType) {
    return ArtsHintBuilder(
        LLVM::UndefOp::create(builder, loc, descriptorType).getResult());
  }

  void setRoute(OpBuilder &builder, Location loc, Value route) {
    setPtr(builder, loc, kRouteField, route);
  }

  void setArtsId(OpBuilder &builder, Location loc, Value artsId) {
    setPtr(builder, loc, kArtsIdField, artsId);
  }

private:
  static constexpr unsigned kRouteField = 0;
  static constexpr unsigned kArtsIdField = 1;
};

///===----------------------------------------------------------------------===///
/// Constants
///===----------------------------------------------------------------------===///

static constexpr int32_t kArtsDepFlagPreserveShape = 1 << 1;

///===----------------------------------------------------------------------===///
/// Helper Functions
///===----------------------------------------------------------------------===///

SmallVector<Value, 4>
materializeStaticDbOuterShape(Value handle, ArtsCodegen *AC, Location loc);

SmallVector<Value, 4> resolveSourceOuterSizes(Value sourceGuid, Value sourcePtr,
                                              ArtsCodegen *AC, Location loc);

SmallVector<Value, 4> resolveOuterSizesForGuid(Value dbGuid, ArtsCodegen *AC,
                                               Location loc);

Value buildArtsHintMemref(ArtsCodegen *AC, Value route, Value artsId,
                          Location loc);

///===----------------------------------------------------------------------===///
/// Pattern Population
///===----------------------------------------------------------------------===///

void populateRuntimePatterns(RewritePatternSet &patterns, ArtsCodegen *AC);
void populateRtToLLVMPatterns(RewritePatternSet &patterns, ArtsCodegen *AC);
void populateDbPatterns(RewritePatternSet &patterns, ArtsCodegen *AC);
void populateOtherPatterns(RewritePatternSet &patterns, ArtsCodegen *AC);

} // namespace mlir::carts::arts_rt::convert_arts_rt_to_llvm

#endif // CARTS_DIALECT_ARTS_RT_CONVERSION_ARTSRTTOLLVM_INTERNAL_H
