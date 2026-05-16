///==========================================================================///
/// File: EdgeBase.h
///
/// Abstract base class for graph edges.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_BASE_EDGEBASE_H
#define ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_BASE_EDGEBASE_H

namespace mlir {
namespace carts::arts {

class NodeBase;

/// Abstract base class for all graph edges.
class EdgeBase {
public:
  virtual ~EdgeBase() = default;

  /// Get source and destination nodes.
  virtual NodeBase *getFrom() const = 0;
  virtual NodeBase *getTo() const = 0;
};

} // namespace carts::arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_BASE_EDGEBASE_H
