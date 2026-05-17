///==========================================================================///
/// File: EdtPtrRematerializationPass.cpp
///
/// Rematerializes pointer-producing ops at EDT use sites to avoid carrying
/// unnecessary pointer SSA values across rewritten EDT boundaries.
///
/// Example:
///   Before:
///     %p = ... ; arts.edt (...) { use %p }
///
///   After:
///     arts.edt (...) { %p' = rematerialized(...); use %p' }
///==========================================================================///

/// Dialects
#define GEN_PASS_DEF_EDTPTRREMATERIALIZATION
#include "carts/passes/Passes.h"
#include "carts/passes/Passes.h.inc"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"

/// Arts
#include "carts/dialect/arts/IR/ArtsDialect.h"
#include "carts/dialect/arts/Transforms/edt/EdtPtrRematerialization.h"

/// LLVM
#include "llvm/ADT/SmallVector.h"

#include "carts/utils/Debug.h"
ARTS_DEBUG_SETUP(edt_ptr_rematerialization);

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

namespace {
struct EdtPtrRematerializationPass
    : public impl::EdtPtrRematerializationBase<EdtPtrRematerializationPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

void EdtPtrRematerializationPass::runOnOperation() {
  ModuleOp module = getOperation();
  ARTS_DEBUG_HEADER(EdtPtrRematerializationPass);
  ARTS_DEBUG_REGION(module.dump(););

  /// Collect all EDT operations
  SmallVector<arts::EdtOp, 4> edtOps;
  module.walk([&](arts::EdtOp edtOp) { edtOps.push_back(edtOp); });

  ARTS_DEBUG("Found " << edtOps.size() << " EDT operations to process");

  /// Walk through all EdtOp instances in the module.
  for (arts::EdtOp edtOp : edtOps)
    rematerializePointersInEdt(edtOp);

  ARTS_DEBUG("Processed " << edtOps.size() << " EDT operations");
  ARTS_DEBUG_FOOTER(EdtPtrRematerializationPass);
  ARTS_DEBUG_REGION(module.dump(););
}

////===----------------------------------------------------------------------===////
/// Pass creation
////===----------------------------------------------------------------------===////
namespace mlir {
namespace carts::arts {
std::unique_ptr<Pass> createEdtPtrRematerializationPass() {
  return std::make_unique<EdtPtrRematerializationPass>();
}

} // namespace carts::arts
} // namespace mlir
