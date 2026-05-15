///==========================================================================///
/// File: CodirBoundaryPlaceholders.cpp
///
/// Registered placeholder passes for the target CODIR boundary. These keep the
/// pass surface build-visible while the active pipeline continues to use the
/// direct SDE-to-ARTS compatibility conversion.
///==========================================================================///

#include "carts/dialect/codir/Transforms/Passes.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::carts::codir {
#define GEN_PASS_DEF_CONVERTCODIRTOARTS
#define GEN_PASS_DEF_CONVERTSDETOCODIR
#define GEN_PASS_DEF_VERIFYCODIR
#include "carts/dialect/codir/Transforms/Passes.h.inc"
} // namespace mlir::carts::codir

using namespace mlir;
using namespace mlir::carts;

namespace {

static bool isCodirDependencyType(Type type) {
  return isa<MemRefType, UnrankedMemRefType>(type);
}

static bool isCodirScalarParamType(Type type) {
  return type.isIntOrIndexOrFloat();
}

static codir::CodirAccessMode
convertAccessMode(sde::SdeAccessMode mode) {
  switch (mode) {
  case sde::SdeAccessMode::read:
    return codir::CodirAccessMode::read;
  case sde::SdeAccessMode::write:
    return codir::CodirAccessMode::write;
  case sde::SdeAccessMode::readwrite:
    return codir::CodirAccessMode::readwrite;
  }
  return codir::CodirAccessMode::readwrite;
}

static SmallVector<OpFoldResult> asIndexFoldResults(ValueRange values,
                                                    OpBuilder &builder) {
  SmallVector<OpFoldResult> results;
  results.reserve(values.size());
  for (Value value : values) {
    APInt constant;
    if (matchPattern(value, m_ConstantInt(&constant))) {
      results.push_back(builder.getIndexAttr(constant.getSExtValue()));
      continue;
    }
    results.push_back(value);
  }
  return results;
}

static LogicalResult materializeCodirDeps(sde::SdeCuCodeletOp sdeCodelet,
                                          OpBuilder &builder,
                                          SmallVectorImpl<Value> &deps,
                                          SmallVectorImpl<Attribute> &modes) {
  deps.clear();
  deps.reserve(sdeCodelet.getTokens().size());
  modes.clear();
  modes.reserve(sdeCodelet.getTokens().size());

  for (auto [index, token] : llvm::enumerate(sdeCodelet.getTokens())) {
    auto tokenOp = token.getDefiningOp<sde::SdeMuTokenOp>();
    if (!tokenOp)
      return sdeCodelet.emitOpError()
             << "convert-sde-to-codir requires token operand #" << index
             << " to be defined by sde.mu_token";

    auto tokenType = dyn_cast<sde::TokenType>(token.getType());
    if (!tokenType)
      return sdeCodelet.emitOpError()
             << "convert-sde-to-codir requires token operand #" << index
             << " to have !sde.token type";

    if (tokenOp.getOffsets().empty() && tokenOp.getSizes().empty()) {
      if (tokenType.getSliceType() != tokenOp.getSource().getType()) {
        return sdeCodelet.emitOpError()
               << "convert-sde-to-codir requires whole-storage sde.mu_token "
                  "operand #"
               << index << " to have source type matching token slice type";
      }
      deps.push_back(tokenOp.getSource());
      modes.push_back(codir::CodirAccessModeAttr::get(
          builder.getContext(), convertAccessMode(tokenOp.getMode())));
      continue;
    }

    SmallVector<OpFoldResult> offsets =
        asIndexFoldResults(tokenOp.getOffsets(), builder);
    SmallVector<OpFoldResult> sizes =
        asIndexFoldResults(tokenOp.getSizes(), builder);
    SmallVector<OpFoldResult> strides(offsets.size(),
                                      builder.getIndexAttr(1));
    auto subview = memref::SubViewOp::create(
        builder, tokenOp.getLoc(), tokenType.getSliceType(), tokenOp.getSource(),
        offsets, sizes, strides);
    deps.push_back(subview.getResult());
    modes.push_back(codir::CodirAccessModeAttr::get(
        builder.getContext(), convertAccessMode(tokenOp.getMode())));
  }

  return success();
}

static void replaceSdeYieldWithCodirYield(codir::CodeletOp codelet) {
  Block &body = codelet.getBody().front();
  auto sdeYield = dyn_cast_or_null<sde::SdeYieldOp>(body.getTerminator());
  if (!sdeYield)
    return;

  OpBuilder builder(sdeYield);
  codir::YieldOp::create(builder, sdeYield.getLoc(),
                                sdeYield.getValues());
  sdeYield.erase();
}

struct ConvertSdeToCodirPass
    : public codir::impl::ConvertSdeToCodirBase<
          ConvertSdeToCodirPass> {
  void runOnOperation() override {
    SmallVector<sde::SdeCuCodeletOp> codelets;
    getOperation().walk(
        [&](sde::SdeCuCodeletOp codelet) { codelets.push_back(codelet); });

    for (sde::SdeCuCodeletOp sdeCodelet : codelets) {
      SmallVector<Value> deps;
      SmallVector<Attribute> depModes;
      OpBuilder builder(sdeCodelet);
      if (failed(materializeCodirDeps(sdeCodelet, builder, deps, depModes))) {
        signalPassFailure();
        return;
      }

      SmallVector<Value> params(sdeCodelet.getCaptures().begin(),
                                sdeCodelet.getCaptures().end());
      auto codirCodelet = codir::CodeletOp::create(
          builder, sdeCodelet.getLoc(), builder.getArrayAttr(depModes), deps,
          params);
      codirCodelet.getBody().takeBody(sdeCodelet.getBody());
      replaceSdeYieldWithCodirYield(codirCodelet);
      sdeCodelet.erase();
    }
  }
};

struct VerifyCodirPass
    : public codir::impl::VerifyCodirBase<VerifyCodirPass> {
  void runOnOperation() override {
    bool failed = false;
    getOperation().walk([&](codir::CodeletOp codelet) {
      auto depModes = codelet->getAttrOfType<ArrayAttr>("dep_modes");
      if (!codelet.getDeps().empty() && !depModes) {
        codelet.emitOpError()
            << "expects dep_modes attribute when dependency operands are "
               "present";
        failed = true;
      }
      if (depModes) {
        if (depModes.size() != codelet.getDeps().size()) {
          codelet.emitOpError()
              << "expects dep_modes entry count (" << depModes.size()
              << ") to match dependency operand count ("
              << codelet.getDeps().size() << ")";
          failed = true;
        }
        for (auto [index, attr] : llvm::enumerate(depModes)) {
          if (isa<codir::CodirAccessModeAttr>(attr))
            continue;
          codelet.emitOpError()
              << "dep_modes entry #" << index
              << " must be a CODIR access_mode attribute, got " << attr;
          failed = true;
        }
      }

      for (auto [index, dep] : llvm::enumerate(codelet.getDeps())) {
        if (isCodirDependencyType(dep.getType()))
          continue;
        codelet.emitOpError()
            << "CODIR dependency #" << index << " must be a memref value, got "
            << dep.getType();
        failed = true;
      }

      for (auto [index, param] : llvm::enumerate(codelet.getParams())) {
        if (isCodirScalarParamType(param.getType()))
          continue;
        codelet.emitOpError()
            << "CODIR parameter #" << index
            << " must be an integer, index, or float scalar; got "
            << param.getType();
        failed = true;
      }

      if (codelet.getBody().empty()) {
        codelet.emitOpError() << "expects body to contain a single block";
        failed = true;
        return;
      }

      Block &body = codelet.getBody().front();
      unsigned numDeps = codelet.getDeps().size();
      unsigned numParams = codelet.getParams().size();
      unsigned expectedArgs = numDeps + numParams;
      if (body.getNumArguments() != expectedArgs) {
        codelet.emitOpError()
            << "expects " << expectedArgs << " CODIR block argument(s) ("
            << numDeps << " dep + " << numParams << " param); got "
            << body.getNumArguments();
        failed = true;
      } else {
        for (auto [index, dep] : llvm::enumerate(codelet.getDeps())) {
          Type argType = body.getArgument(index).getType();
          if (argType == dep.getType())
            continue;
          codelet.emitOpError()
              << "dep block argument #" << index << " type (" << argType
              << ") does not match dep operand type (" << dep.getType()
              << ")";
          failed = true;
        }

        for (auto [index, param] : llvm::enumerate(codelet.getParams())) {
          unsigned argIndex = numDeps + index;
          Type argType = body.getArgument(argIndex).getType();
          if (argType == param.getType())
            continue;
          codelet.emitOpError()
              << "param block argument #" << index << " type (" << argType
              << ") does not match param operand type (" << param.getType()
              << ")";
          failed = true;
        }
      }

      auto yield =
          dyn_cast_or_null<codir::YieldOp>(body.getTerminator());
      if (!yield) {
        codelet.emitOpError() << "expects body to terminate with codir.yield";
        failed = true;
        return;
      }

      for (auto [index, yielded] : llvm::enumerate(yield.getResults())) {
        if (isCodirScalarParamType(yielded.getType()))
          continue;
        codelet.emitOpError()
            << "CODIR yield operand #" << index
            << " must be an integer, index, or float scalar; got "
            << yielded.getType();
        failed = true;
      }
    });

    if (failed)
      signalPassFailure();
  }
};

struct ConvertCodirToArtsPass
    : public codir::impl::ConvertCodirToArtsBase<
          ConvertCodirToArtsPass> {
  void runOnOperation() override {}
};

} // namespace

std::unique_ptr<Pass> mlir::carts::codir::createConvertSdeToCodirPass() {
  return std::make_unique<ConvertSdeToCodirPass>();
}

std::unique_ptr<Pass> mlir::carts::codir::createVerifyCodirPass() {
  return std::make_unique<VerifyCodirPass>();
}

std::unique_ptr<Pass> mlir::carts::codir::createConvertCodirToArtsPass() {
  return std::make_unique<ConvertCodirToArtsPass>();
}
