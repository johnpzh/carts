///==========================================================================///
/// File: EdtInvariantCodeMotion.cpp
///
/// Hoists loop-invariant computations inside EDT regions to reduce repeated
/// work and improve downstream simplification opportunities.
///
/// Example:
///   Before:
///     scf.for %i = ... { %c = arith.addi %N, %M ... use %c }
///
///   After:
///     %c = arith.addi %N, %M
///     scf.for %i = ... { ... use %c }
///==========================================================================///

/// Dialects
#define GEN_PASS_DEF_EDTICM
#include "carts/passes/Passes.h"
#include "carts/passes/Passes.h.inc"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"

/// Arts
#include "carts/Dialect.h"

/// Others
#include "carts/dialect/arts/Transforms/edt/EdtICM.h"

/// LLVM support
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "carts/utils/Debug.h"
ARTS_DEBUG_SETUP(edt_icm);

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

namespace {
struct EdtICMPass : public impl::EdtICMBase<EdtICMPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

void EdtICMPass::runOnOperation() {
  ModuleOp module = getOperation();
  ARTS_INFO_HEADER(EdtICMPass);
  ARTS_DEBUG_REGION(module.dump(););

  bool changed = false;
  /// Walk through all EdtOp instances in the module.
  module.walk([&](arts::EdtOp edtOp) {
    ARTS_DEBUG_TYPE("Processing EDT:\n" << edtOp);

    /// Use the new function to move invariant code out of this EDT.
    auto movedCount = moveEdtInvariantCode(edtOp);
    if (movedCount > 0)
      changed = true;

    ARTS_INFO("Moved " << movedCount << " operations out of EDT.");
  });

  /// if (!changed)
  ///  markAllAnalysesPreserved();

  ARTS_INFO_FOOTER(EdtInvariantCodeMotionPass);
  ARTS_DEBUG_REGION(module.dump(););
}

///===----------------------------------------------------------------------===///
/// Pass creation
///===----------------------------------------------------------------------===///
namespace mlir {
namespace carts::arts {
std::unique_ptr<Pass> createEdtICMPass() {
  return std::make_unique<EdtICMPass>();
}

} // namespace carts::arts
} // namespace mlir
