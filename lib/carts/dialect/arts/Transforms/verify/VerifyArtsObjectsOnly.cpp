///==========================================================================///
/// File: VerifyArtsObjectsOnly.cpp
///
/// Verification pass for the target CODIR-to-ARTS boundary contract.
///==========================================================================///

#include "carts/dialect/arts/IR/ArtsDialect.h"
#include "carts/dialect/arts/Utils/DbUtils.h"
#include "carts/dialect/sde/IR/SdeDialect.h"
#define GEN_PASS_DEF_VERIFYARTSOBJECTSONLY
#include "carts/passes/Passes.h"
#include "carts/passes/Passes.h.inc"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

namespace {

constexpr llvm::StringLiteral kKeepHostOpenMP = "sde.keep_host_openmp";

static bool isInsideHostOpenMPIsland(Operation *op) {
  for (Operation *cur = op; cur; cur = cur->getParentOp())
    if (cur->hasAttr(kKeepHostOpenMP))
      return true;
  return false;
}

static bool isSmallReadOnlyCoarseDep(Value dep, DbAllocOp alloc) {
  auto acquire = dep.getDefiningOp<DbAcquireOp>();
  return acquire && acquire.getMode() == ArtsMode::in &&
         DbUtils::isSmallCoarseUserDataDb(alloc);
}

static void verifyDistributedDbDeps(EdtOp edt, bool &found) {
  if (!edt || edt.getConcurrency() != EdtConcurrency::internode)
    return;

  llvm::SmallPtrSet<Operation *, 4> reported;
  for (Value dep : edt.getDependencies()) {
    auto alloc =
        dyn_cast_or_null<DbAllocOp>(DbUtils::getUnderlyingDbAlloc(dep));
    if (!DbUtils::isCoarseUserDataDb(alloc))
      continue;
    if (isSmallReadOnlyCoarseDep(dep, alloc))
      continue;
    if (!reported.insert(alloc.getOperation()).second)
      continue;

    InFlightDiagnostic diag = edt.emitError()
        << "internode ARTS task depends on a coarse single-block aggregate "
           "user DB";
    diag.attachNote(alloc.getLoc())
        << "coarse DB allocation feeding the distributed task";
    diag.attachNote(edt.getLoc())
        << "SDE/CODIR must materialize block DB storage before ARTS "
           "distributed execution; CreateDbs is only a coarse raw-memref "
           "fallback";
    found = true;
  }
}

static LogicalResult verifyArtsObjectsOnly(ModuleOp module) {
  auto *sdeDialect =
      module->getContext()->getLoadedDialect<sde::CartsSdeDialect>();
  bool found = false;

  module.walk([&](Operation *op) {
    if (sdeDialect && op->getDialect() == sdeDialect) {
      op->emitError() << "SDE operation '" << op->getName()
                      << "' remains after the CODIR-to-ARTS boundary";
      found = true;
    }

    if (auto edt = dyn_cast<EdtOp>(op))
      verifyDistributedDbDeps(edt, found);

    if (op->getDialect() && op->getDialect()->getNamespace() == "omp") {
      if (isInsideHostOpenMPIsland(op))
        return;
      op->emitError() << "OpenMP operation '" << op->getName()
                      << "' remains after the CODIR-to-ARTS boundary";
      found = true;
    }

    if (isa<scf::ParallelOp>(op)) {
      if (isInsideHostOpenMPIsland(op))
        return;
      op->emitError()
          << "scf.parallel remains after the CODIR-to-ARTS boundary; "
             "parallel work must be materialized as ARTS objects or marked as "
             "host OpenMP fallback";
      found = true;
    }
  });

  return failure(found);
}

struct VerifyArtsObjectsOnlyPass
    : public impl::VerifyArtsObjectsOnlyBase<VerifyArtsObjectsOnlyPass> {
  VerifyArtsObjectsOnlyPass() = default;

  void runOnOperation() override {
    if (failed(verifyArtsObjectsOnly(getOperation())))
      signalPassFailure();
  }
};

} /// namespace

std::unique_ptr<Pass> mlir::carts::arts::createVerifyArtsObjectsOnlyPass() {
  return std::make_unique<VerifyArtsObjectsOnlyPass>();
}
