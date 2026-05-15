///==========================================================================///
/// File: EdgeBase.h
///
/// Abstract base class for graph edges.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_BASE_EDGEBASE_H
#define ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_BASE_EDGEBASE_H

#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace arts {

class NodeBase;

/// Abstract base class for all edges (e.g., dep edges, alloc edges).
/// Provides common interface for from/to, type, and printing.
class EdgeBase {
public:
  virtual ~EdgeBase() = default;

  enum class EdgeKind { Dep };

  /// Get source and destination nodes.
  virtual NodeBase *getFrom() const = 0;
  virtual NodeBase *getTo() const = 0;

  /// Strongly-typed edge kind for internal logic.
  virtual EdgeKind getKind() const = 0;

  /// Human-readable label for debugging/printing only
  virtual llvm::StringRef getType() const = 0;

  /// Print edge details.
  virtual void print(llvm::raw_ostream &os) const = 0;
};

} // namespace arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_BASE_EDGEBASE_H