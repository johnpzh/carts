///==========================================================================///
/// File: LaunchPolicyUtils.h
///
/// ARTS-owned helpers for binding generic launch intent to ARTS topology.
///==========================================================================///

#ifndef CARTS_DIALECT_ARTS_UTILS_LAUNCHPOLICYUTILS_H
#define CARTS_DIALECT_ARTS_UTILS_LAUNCHPOLICYUTILS_H

#include "carts/dialect/arts/IR/ArtsDialect.h"
#include "carts/utils/OperationAttributes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include <optional>

namespace mlir::carts::arts {

struct ArtsLaunchPolicy {
  EdtConcurrency concurrency = EdtConcurrency::intranode;
  Value route;
};

inline bool hasArtsInterNodeRuntime(ModuleOp module) {
  std::optional<int64_t> totalNodes = getRuntimeTotalNodes(module);
  return totalNodes && *totalNodes > 1;
}

inline ArtsLaunchPolicy resolveArtsOrdinalLaunchPolicy(ModuleOp module,
                                                       Value ordinal,
                                                       OpBuilder &builder,
                                                       Location loc) {
  ArtsLaunchPolicy policy;
  if (!module || !hasArtsInterNodeRuntime(module) || !ordinal)
    return policy;

  policy.concurrency = EdtConcurrency::internode;
  Value ordinalI32 =
      arith::IndexCastOp::create(builder, loc, builder.getI32Type(), ordinal);
  auto totalNodes =
      RuntimeQueryOp::create(builder, loc, RuntimeQueryKind::totalNodes);
  policy.route =
      arith::RemUIOp::create(builder, loc, ordinalI32, totalNodes.getResult());
  return policy;
}

inline ArtsLaunchPolicy
resolveArtsLaunchPolicy(ModuleOp module, scf::ForOp dispatchLoop,
                        bool hasDistributedLaunchStoragePlan,
                        OpBuilder &builder, Location loc) {
  ArtsLaunchPolicy policy;
  if (!module || !hasArtsInterNodeRuntime(module) ||
      !hasDistributedLaunchStoragePlan || !dispatchLoop)
    return policy;

  policy.concurrency = EdtConcurrency::internode;
  Value relative =
      arith::SubIOp::create(builder, loc, dispatchLoop.getInductionVar(),
                            dispatchLoop.getLowerBound());
  Value ordinal =
      arith::DivUIOp::create(builder, loc, relative, dispatchLoop.getStep());
  return resolveArtsOrdinalLaunchPolicy(module, ordinal, builder, loc);
}

} // namespace mlir::carts::arts

#endif // CARTS_DIALECT_ARTS_UTILS_LAUNCHPOLICYUTILS_H
