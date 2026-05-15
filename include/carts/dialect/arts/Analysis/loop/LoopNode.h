///==========================================================================///
/// File: LoopNode.h
///
/// LoopNode represents a loop operation with integrated metadata.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_ANALYSIS_LOOP_LOOPNODE_H
#define ARTS_DIALECT_CORE_ANALYSIS_LOOP_LOOPNODE_H

#include "carts/dialect/arts/Analysis/db/DbAnalysis.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <string>

namespace mlir {
namespace carts::arts {

///==========================================================================///
/// LoopNode - Represents a loop operation with metadata
///==========================================================================///
class LoopAnalysis;
struct DbAcquirePartitionFacts;
class LoopNode {
public:
  /// Constructor: automatically imports metadata from operation
  explicit LoopNode(Operation *loopOp, LoopAnalysis *loopAnalysis = nullptr);

  /// Get the underlying loop operation (affine.for, scf.for, scf.parallel)
  Operation *getLoopOp() const { return loopOp; }

  StringRef getHierId() const { return hierId; }
  void setHierId(std::string id) { hierId = id; }
  void print(llvm::raw_ostream &os) const;
  Operation *getOp() const { return loopOp; }

  //===--------------------------------------------------------------------===//
  /// Induction Variable Analysis
  //===--------------------------------------------------------------------===//

  /// Get the induction variable for this loop
  Value getInductionVar() const;

  /// Utility functions
  bool dependsOnInductionVar(Value v);
  bool dependsOnInductionVarNormalized(Value v);
  bool isValueLoopInvariant(Value v);
  static bool dependsOnLoopInit(Value value, Value base);
  static bool dependsOnLoopInitNormalized(Value value, Value base);

  /// Analyzed index expression relative to induction variable.
  /// Represents: index = iv * multiplier + offset
  struct IVExpr {
    bool dependsOnIV = false;
    std::optional<int64_t> offset;     /// e.g., -1 for i-1, +1 for i+1
    std::optional<int64_t> multiplier; /// e.g., 2 for 2*i
    bool isAnalyzable() const { return dependsOnIV && offset.has_value(); }
  };

  /// Analyze an index expression to extract IV relationship.
  /// Returns IVExpr with offset/multiplier if analyzable.
  IVExpr analyzeIndexExpr(Value index);
  std::optional<int64_t> getLowerBoundConstant() const;
  std::optional<int64_t> getUpperBoundConstant() const;
  std::optional<int64_t> getStepConstant() const;
  std::optional<int64_t> estimateStaticPerfectNestedWork(int64_t cap = 8) const;
  int getNestingDepth() const;
  void clearIVCache() { ivDepCache.clear(); }

  //===--------------------------------------------------------------------===//
  /// Raw Value Accessors
  //===--------------------------------------------------------------------===//

  /// Get the lower bound value for this loop.
  Value getLowerBound() const;

  /// Get the upper bound value for this loop.
  Value getUpperBound() const;

  /// Get the step value for this loop.
  Value getStep() const;

  //===--------------------------------------------------------------------===//
  /// Loop Classification Methods
  //===--------------------------------------------------------------------===//

  /// Check if this loop is innermost (contains no nested loops of same type).
  bool isInnermostLoop() const;

  /// Check if this loop contains EDT operations (worker loop in ARTS).
  bool hasEdt() const;

  /// Loop-scoped DB partition analysis, proxied through LoopAnalysis.
  bool hasDistributedDbContract() const;
  bool hasPeerInferredPartitionDims() const;

  /// Check if two loops have compatible bounds for fusion (lower, upper, step).
  static bool haveCompatibleBounds(LoopNode *a, LoopNode *b);

private:
  Operation *loopOp;
  LoopAnalysis *loopAnalysis = nullptr;
  std::string hierId;

  /// Cache for IV dependency analysis to avoid repeated traversals
  DenseMap<Value, bool> ivDepCache;
};

} // namespace carts::arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_ANALYSIS_LOOP_LOOPNODE_H
