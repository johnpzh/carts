///==========================================================================///
/// File: Dialect.cpp
/// Defines the Arts dialect and the operations within it.
///==========================================================================///

#include "arts/utils/OperationAttributes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include <cassert>

#include "arts/Dialect.h"
#include "arts/dialect/core/Analysis/db/DbAnalysis.h"
#include "arts/utils/DbUtils.h"
#include "arts/utils/StencilAttributes.h"
#include "arts/utils/Utils.h"
#include "arts/utils/ValueAnalysis.h"

using namespace mlir;
using namespace mlir::arts;

void ArtsDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "arts/dialect/core/IR/Ops.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "arts/dialect/core/IR/OpsTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "arts/dialect/core/IR/OpsAttributes.cpp.inc"
      >();
}

#include "arts/dialect/core/IR/OpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "arts/dialect/core/IR/Ops.cpp.inc"

namespace {
struct FoldDbDimFromDbOps : public OpRewritePattern<DbDimOp> {
  using OpRewritePattern<DbDimOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(DbDimOp op,
                                PatternRewriter &rewriter) const override {
    auto cstIdx = op.getDim().getDefiningOp<arith::ConstantIndexOp>();
    if (!cstIdx)
      return failure();
    int64_t idx = cast<IntegerAttr>(cstIdx.getValue()).getInt();

    auto defOp = op.getSource().getDefiningOp();
    if (defOp && (isa<DbAllocOp>(defOp) || isa<DbAcquireOp>(defOp))) {
      auto sizes = DbUtils::getSizesFromDb(defOp);
      if (static_cast<int64_t>(sizes.size()) > idx) {
        rewriter.replaceOp(op, sizes[idx]);
        return success();
      }
    }
    /// Fallback: query with memref.dim if source is not a DB op
    rewriter.replaceOpWithNewOp<memref::DimOp>(op, op.getSource(), idx);
    return success();
  }
};

struct RemoveUnusedDbAcquire : public OpRewritePattern<DbAcquireOp> {
  using OpRewritePattern<DbAcquireOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DbAcquireOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getGuid().use_empty() && op.getPtr().use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

struct FoldKnownRuntimeQuery : public OpRewritePattern<RuntimeQueryOp> {
  using OpRewritePattern<RuntimeQueryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(RuntimeQueryOp op,
                                PatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    std::optional<int64_t> foldedValue;
    switch (op.getKind()) {
    case RuntimeQueryKind::totalWorkers:
      /// Keep worker count dynamic unless the module explicitly requests
      /// compile-time worker folding (e.g., benchmarking reproducibility).
      if (!getRuntimeStaticWorkers(module))
        return failure();
      foldedValue = getRuntimeTotalWorkers(module);
      break;
    case RuntimeQueryKind::totalNodes:
      foldedValue = getRuntimeTotalNodes(module);
      break;
    default:
      return failure();
    }

    if (!foldedValue || *foldedValue <= 0)
      return failure();

    Type resultType = op.getType();
    if (isa<IndexType>(resultType)) {
      rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, *foldedValue);
      return success();
    }
    if (auto intTy = dyn_cast<IntegerType>(resultType)) {
      rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(op, *foldedValue,
                                                        intTy.getWidth());
      return success();
    }
    return failure();
  }
};
} // namespace

void DbDimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add<FoldDbDimFromDbOps>(context);
}

void DbAcquireOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<RemoveUnusedDbAcquire>(context);
}

void RuntimeQueryOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.add<FoldKnownRuntimeQuery>(context);
}

bool isArtsRegion(Operation *op) { return isa<EdtOp>(op) || isa<EpochOp>(op); }
bool isArtsOp(Operation *op) {
  return isArtsRegion(op) ||
         isa<arts::BarrierOp, arts::AllocOp, arts::DbAllocOp, arts::DbAcquireOp,
             arts::DbReleaseOp, arts::DbFreeOp, arts::DbControlOp,
             arts::RuntimeQueryOp>(op);
}

/// Arts Dialect Types
#define GET_TYPEDEF_CLASSES
#include "arts/dialect/core/IR/OpsTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "arts/dialect/core/IR/OpsAttributes.cpp.inc"

#include "arts/dialect/core/IR/OpsEnums.cpp.inc"

class UndefToLLVM final : public OpRewritePattern<UndefOp> {
public:
  using OpRewritePattern<UndefOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(UndefOp uop,
                                PatternRewriter &rewriter) const override {
    auto ty = uop.getResult().getType();
    if (!LLVM::isCompatibleType(ty))
      return failure();
    rewriter.replaceOpWithNewOp<LLVM::UndefOp>(uop, ty);
    return success();
  }
};

void UndefOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.insert<UndefToLLVM>(context);
}

SmallVector<Value> mlir::arts::EdtOp::getDependenciesAsVector() {
  SmallVector<Value> deps(getDependencies().begin(), getDependencies().end());
  return deps;
}

void mlir::arts::EdtOp::setDependencies(ValueRange newDeps) {
  Operation *op = getOperation();
  SmallVector<Value> operands;

  if (op->getNumOperands() > 0)
    operands.push_back(op->getOperand(0));

  operands.append(newDeps.begin(), newDeps.end());
  op->setOperands(operands);
}

void mlir::arts::EdtOp::appendDependency(Value dep) {
  SmallVector<Value> deps(getDependencies().begin(), getDependencies().end());
  deps.push_back(dep);
  setDependencies(deps);
}

LogicalResult EdtOp::verify() {
  if (getNoVerifyAttr())
    return success();

  Block &block = getBody().front();
  auto blockArgs = block.getArguments();
  auto deps = getDependenciesAsVector();

  if (blockArgs.size() != deps.size()) {
    return emitOpError("block arguments (")
           << blockArgs.size() << ") must match dependencies (" << deps.size()
           << ")";
  }

  for (Value dep : deps) {
    auto dbAcquireOp = dep.getDefiningOp<DbAcquireOp>();
    if (!dbAcquireOp) {
      return emitOpError("dependency must be a result of DbAcquireOp, got: ")
             << dep;
    }
  }

  DenseSet<Value> externalValues;
  for (Value dep : deps)
    externalValues.insert(dep);

  auto walkResult = getBody().walk([&](Operation *op) {
    if (op == &block.front())
      return WalkResult::advance();

    if (isa<EdtOp>(op))
      return WalkResult::skip();

    for (Value operand : op->getOperands()) {
      if (!isa<MemRefType>(operand.getType()))
        continue;

      if (llvm::is_contained(blockArgs, operand)) {
        DbAcquireOp underlyingAcquire;
        if (auto *rawDb = DbUtils::getUnderlyingDb(operand))
          underlyingAcquire = dyn_cast<DbAcquireOp>(rawDb);
        if (!underlyingAcquire) {
          op->emitOpError("EDT region uses block argument '")
              << operand << "' as a DbAcquire value.";
          return WalkResult::interrupt();
        }
        continue;
      }

      auto definingOp = operand.getDefiningOp();
      if (!definingOp)
        continue;

      if (getBody().isAncestor(definingOp->getParentRegion()))
        continue;

      /// Allow external memref allocations. Stack allocas are sunk into EDTs
      /// later to keep task-local buffers private; heap memrefs are captured
      /// by EdtEnvManager and packed through paramv.
      if (operand.getDefiningOp<memref::AllocaOp>() ||
          operand.getDefiningOp<memref::AllocOp>())
        continue;

      /// Allow external DbAllocOp results; EdtEnvManager already captures
      /// these as dbHandles and packs them through paramv.
      if (operand.getDefiningOp<DbAllocOp>())
        continue;

      /// DbAcquire source handles may intentionally reference outer-scope
      /// datablock handles to derive nested views inside EDTs.
      if (auto dbAcquire = dyn_cast<DbAcquireOp>(op)) {
        if (operand == dbAcquire.getSourceGuid() ||
            operand == dbAcquire.getSourcePtr())
          continue;
      }

      /// CPS chain continuations clone epoch bodies from the original loop,
      /// which may reference external values (memref.alloc, pointer2memref,
      /// etc.). These are captured by EdtEnvManager during EdtLowering.
      if ((*this)->hasAttr(AttrNames::Operation::CPSChainId))
        continue;

      if (externalValues.contains(operand)) {
        op->emitOpError("EDT region uses external DbAcquire value '")
            << operand << "' directly instead of DbAcquire block argument. ";
        return WalkResult::interrupt();
      }

      op->emitOpError("EDT region uses external value '")
          << operand << "' that is not a block argument or dependency.\n"
          << *definingOp << "\n"
          << *op << "\n";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted())
    return failure();

  return success();
}

void EdtOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  auto dependencies = getDependenciesAsVector();

  if (dependencies.empty()) {
    effects.emplace_back(MemoryEffects::Read::get(),
                         ::mlir::SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(),
                         ::mlir::SideEffects::DefaultResource::get());
    return;
  }
}

void DbReleaseOp::build(OpBuilder &builder, OperationState &state,
                        Value source) {
  state.addOperands(source);
}

void DbDimOp::build(OpBuilder &builder, OperationState &state, Value source,
                    int64_t dim) {
  state.addOperands(source);
  Value c = arts::createConstantIndex(builder, state.location, dim);
  state.addOperands(c);
  state.addTypes(builder.getIndexType());
}

void DbControlOp::build(OpBuilder &builder, OperationState &state,
                        ArtsMode mode, Value ptr, SmallVector<Value> indices,
                        SmallVector<Value> offsets, SmallVector<Value> sizes) {
  auto subviewType =
      MemRefType::get({ShapedType::kDynamic}, builder.getI64Type());
  DbControlOp::build(builder, state, subviewType, mode, ptr, indices, offsets,
                     sizes);
}

void EdtOp::build(OpBuilder &builder, OperationState &state, EdtType type,
                  EdtConcurrency concurrency) {
  build(builder, state, type, concurrency, ValueRange{});
}

void EdtOp::build(OpBuilder &builder, OperationState &state, EdtType type,
                  EdtConcurrency concurrency, ValueRange dependencies) {
  /// Default EDT placement stays on the current node unless a pass provides
  /// an explicit route. Using node 0 here would silently bias multi-node
  /// programs toward rank 0 during lowering.
  Value route = createCurrentNodeRoute(builder, state.location);
  build(builder, state, type, concurrency, route, dependencies);
}

void EdtOp::build(OpBuilder &builder, OperationState &state, EdtType type,
                  EdtConcurrency concurrency, Value route,
                  ValueRange dependencies) {
  state.addAttribute("type", EdtTypeAttr::get(builder.getContext(), type));
  state.addAttribute("concurrency", EdtConcurrencyAttr::get(
                                        builder.getContext(), concurrency));
  state.addOperands(route);
  state.addOperands(dependencies);

  Region *bodyRegion = state.addRegion();
  Block *bodyBlock = new Block();
  bodyRegion->push_back(bodyBlock);
}

/// Helper to compute GUID type from sizes
static Type computeGuidType(OpBuilder &builder, ArrayRef<Value> sizes) {
  size_t numDims = sizes.empty() ? 1 : std::max(sizes.size(), size_t(1));
  SmallVector<int64_t> shape(numDims, ShapedType::kDynamic);
  return MemRefType::get(shape, builder.getI64Type());
}

/// Builder-side bundle for DbAllocOp construction so individual overloads don't
/// have to thread a long positional argument list into the shared helper.
struct DbAllocBuildInfo {
  ArtsMode mode;
  Value route;
  DbAllocType allocType;
  DbMode dbMode;
  Type elementType;
  Value address;
  SmallVector<Value> sizes;
  SmallVector<Value> elementSizes;
  Type pointerType;
  PartitionMode partitionMode = PartitionMode::coarse;
};

static void buildDbAllocOpCommon(OpBuilder &builder, OperationState &state,
                                 DbAllocBuildInfo info) {
  /// Auto-compute GUID type to match datablock dimensionality
  Type guidType = computeGuidType(builder, info.sizes);

  if (info.sizes.empty())
    info.sizes.push_back(arts::createOneIndex(builder, state.location));

  if (info.elementSizes.empty())
    info.elementSizes.push_back(arts::createOneIndex(builder, state.location));

  Type ptrType;
  if (info.pointerType) {
    ptrType = info.pointerType;
  } else {
    Type pointerElementType =
        arts::getElementMemRefType(info.elementType, info.elementSizes);
    SmallVector<int64_t> shape;
    shape.assign(info.sizes.size(), ShapedType::kDynamic);
    ptrType = MemRefType::get(shape, pointerElementType);
  }
  state.addTypes({guidType, ptrType});

  ArtsModeAttr modeAttr = ArtsModeAttr::get(builder.getContext(), info.mode);
  DbAllocTypeAttr allocTypeAttr =
      DbAllocTypeAttr::get(builder.getContext(), info.allocType);
  DbModeAttr dbModeAttr = DbModeAttr::get(builder.getContext(), info.dbMode);
  TypeAttr elementTypeAttr = TypeAttr::get(info.elementType);

  state.addAttribute("mode", modeAttr);
  state.addAttribute("allocType", allocTypeAttr);
  state.addAttribute("dbMode", dbModeAttr);
  state.addAttribute("elementType", elementTypeAttr);
  state.addAttribute(
      AttrNames::Operation::PartitionMode,
      PartitionModeAttr::get(builder.getContext(), info.partitionMode));

  state.addOperands(info.route);
  if (info.address)
    state.addOperands(info.address);
  state.addOperands(info.sizes);
  state.addOperands(info.elementSizes);

  state.addAttribute(DbAllocOp::getOperandSegmentSizesAttrName(state.name),
                     builder.getDenseI32ArrayAttr(
                         {1, static_cast<int32_t>(info.address ? 1 : 0),
                          static_cast<int32_t>(info.sizes.size()),
                          static_cast<int32_t>(info.elementSizes.size())}));
}

void DbAllocOp::build(OpBuilder &builder, OperationState &state, ArtsMode mode,
                      Value route, DbAllocType allocType, DbMode dbMode,
                      Type elementType, Value address, SmallVector<Value> sizes,
                      SmallVector<Value> elementSizes,
                      PartitionMode partitionMode) {
  buildDbAllocOpCommon(builder, state,
                       {.mode = mode,
                        .route = route,
                        .allocType = allocType,
                        .dbMode = dbMode,
                        .elementType = elementType,
                        .address = address,
                        .sizes = std::move(sizes),
                        .elementSizes = std::move(elementSizes),
                        .pointerType = Type{},
                        .partitionMode = partitionMode});
}

void DbAllocOp::build(OpBuilder &builder, OperationState &state, ArtsMode mode,
                      Value route, DbAllocType allocType, DbMode dbMode,
                      Type elementType, SmallVector<Value> sizes,
                      SmallVector<Value> elementSizes,
                      PartitionMode partitionMode) {
  buildDbAllocOpCommon(builder, state,
                       {.mode = mode,
                        .route = route,
                        .allocType = allocType,
                        .dbMode = dbMode,
                        .elementType = elementType,
                        .address = Value{},
                        .sizes = std::move(sizes),
                        .elementSizes = std::move(elementSizes),
                        .pointerType = Type{},
                        .partitionMode = partitionMode});
}

void DbAllocOp::build(OpBuilder &builder, OperationState &state, ArtsMode mode,
                      Value route, DbAllocType allocType, DbMode dbMode,
                      Type elementType, Type pointerType,
                      SmallVector<Value> sizes, SmallVector<Value> elementSizes,
                      PartitionMode partitionMode) {
  buildDbAllocOpCommon(builder, state,
                       {.mode = mode,
                        .route = route,
                        .allocType = allocType,
                        .dbMode = dbMode,
                        .elementType = elementType,
                        .address = Value{},
                        .sizes = std::move(sizes),
                        .elementSizes = std::move(elementSizes),
                        .pointerType = pointerType,
                        .partitionMode = partitionMode});
}

MemRefType DbAllocOp::getAllocatedElementType() {
  auto ptrType = cast<MemRefType>(getPtr().getType());
  if (auto innerMemRefType = dyn_cast<MemRefType>(ptrType.getElementType()))
    return innerMemRefType;

  return arts::getElementMemRefType(
      getElementType(),
      SmallVector<Value>(getElementSizes().begin(), getElementSizes().end()));
}

LogicalResult DbAllocOp::verify() {
  if (getElementSizes().empty())
    return emitOpError(
        "elementSizes must be non-empty; use a single size of 1 for scalars");
  return success();
}

bool DbAcquireOp::hasExplicitPartitionHints() {
  return !getPartitionIndices().empty() || !getPartitionOffsets().empty() ||
         !getPartitionSizes().empty();
}

void DbAcquireOp::setPreserveAccessMode(bool preserve) {
  if (preserve) {
    setPreserveAccessModeAttr(PreserveAccessModeAttr::get(getContext()));
    return;
  }
  (*this)->removeAttr(AttrNames::Operation::PreserveAccessMode);
}

void DbAcquireOp::setPreserveDepEdge(bool preserve) {
  if (preserve) {
    setPreserveDepEdgeAttr(PreserveDepEdgeAttr::get(getContext()));
    return;
  }
  (*this)->removeAttr(AttrNames::Operation::PreserveDepEdge);
}

void DbAcquireOp::setExplicitDepContract(bool preserve) {
  setPreserveAccessMode(preserve);
  setPreserveDepEdge(preserve);
}

void DbAcquireOp::setDepPattern(ArtsDepPattern pattern) {
  arts::setDepPattern(getOperation(), pattern);
}

void DbAcquireOp::clearDepPattern() {
  (*this)->removeAttr(AttrNames::Operation::DepPatternAttr);
}

std::optional<ArtsDepPattern> DbAcquireOp::getDepPattern() {
  return arts::getDepPattern(getOperation());
}

void DbAcquireOp::copyPartitionSegmentsFrom(DbAcquireOp source) {
  if (!source)
    return;

  OpBuilder builder(getContext());
  if (auto segments = source.getPartitionIndicesSegments())
    setPartitionIndicesSegmentsAttr(builder.getDenseI32ArrayAttr(*segments));
  if (auto segments = source.getPartitionOffsetsSegments())
    setPartitionOffsetsSegmentsAttr(builder.getDenseI32ArrayAttr(*segments));
  if (auto segments = source.getPartitionSizesSegments())
    setPartitionSizesSegmentsAttr(builder.getDenseI32ArrayAttr(*segments));
  if (auto modes = source.getPartitionEntryModes())
    setPartitionEntryModesAttr(builder.getDenseI32ArrayAttr(*modes));
}

bool DbAcquireOp::hasMultiplePartitionEntries() {
  auto segments = getPartitionIndicesSegments();
  return segments && segments->size() > 1;
}

size_t DbAcquireOp::getNumPartitionEntries() {
  auto segments = getPartitionIndicesSegments();
  if (segments && !segments->empty())
    return segments->size();
  if (!getPartitionIndices().empty() || !getPartitionOffsets().empty() ||
      !getPartitionSizes().empty())
    return 1;
  return 0;
}

PartitionMode DbAcquireOp::getPartitionEntryMode(size_t entryIdx) {
  auto modes = getPartitionEntryModes();
  if (modes && entryIdx < modes->size())
    return static_cast<PartitionMode>((*modes)[entryIdx]);
  return getPartitionModeOr();
}

/// Helper to extract a segment of values from a flat array using segment info
static SmallVector<Value>
getSegmentedValues(std::optional<ArrayRef<int32_t>> segments,
                   OperandRange allValues, size_t entryIdx) {
  if (!segments || segments->empty()) {
    if (entryIdx == 0)
      return SmallVector<Value>(allValues.begin(), allValues.end());
    return {};
  }
  if (entryIdx >= segments->size())
    return {};

  int64_t offset = 0;
  for (size_t i = 0; i < entryIdx; ++i)
    offset += (*segments)[i];

  int64_t size = (*segments)[entryIdx];
  SmallVector<Value> result;
  for (int64_t i = 0; i < size && (offset + i) < (int64_t)allValues.size(); ++i)
    result.push_back(allValues[offset + i]);
  return result;
}

SmallVector<Value> DbAcquireOp::getPartitionIndicesForEntry(size_t entryIdx) {
  return getSegmentedValues(getPartitionIndicesSegments(),
                            getPartitionIndices(), entryIdx);
}

SmallVector<Value> DbAcquireOp::getPartitionOffsetsForEntry(size_t entryIdx) {
  return getSegmentedValues(getPartitionOffsetsSegments(),
                            getPartitionOffsets(), entryIdx);
}

SmallVector<Value> DbAcquireOp::getPartitionSizesForEntry(size_t entryIdx) {
  return getSegmentedValues(getPartitionSizesSegments(), getPartitionSizes(),
                            entryIdx);
}

void DbAcquireOp::setPartitionSegments(ArrayRef<int32_t> indicesSegments,
                                       ArrayRef<int32_t> offsetsSegments,
                                       ArrayRef<int32_t> sizesSegments,
                                       ArrayRef<int32_t> entryModes) {
  OpBuilder builder(getContext());
  if (!indicesSegments.empty())
    setPartitionIndicesSegmentsAttr(
        builder.getDenseI32ArrayAttr(indicesSegments));
  if (!offsetsSegments.empty())
    setPartitionOffsetsSegmentsAttr(
        builder.getDenseI32ArrayAttr(offsetsSegments));
  if (!sizesSegments.empty())
    setPartitionSizesSegmentsAttr(builder.getDenseI32ArrayAttr(sizesSegments));
  if (!entryModes.empty())
    setPartitionEntryModesAttr(builder.getDenseI32ArrayAttr(entryModes));
}

SmallVector<PartitionInfo> DbAcquireOp::getPartitionInfos() {
  SmallVector<PartitionInfo> result;
  size_t numEntries = getNumPartitionEntries();

  for (size_t i = 0; i < numEntries; ++i) {
    PartitionInfo info;
    info.mode = getPartitionEntryMode(i);
    info.indices = getPartitionIndicesForEntry(i);
    info.offsets = getPartitionOffsetsForEntry(i);
    info.sizes = getPartitionSizesForEntry(i);
    result.push_back(info);
  }
  return result;
}

void DbAcquireOp::clearPartitionHints() {
  /// Clear partition operands
  getPartitionIndicesMutable().clear();
  getPartitionOffsetsMutable().clear();
  getPartitionSizesMutable().clear();

  /// Clear partition mode attribute
  if (getPartitionMode().has_value())
    removePartitionModeAttr();

  /// Clear segment attributes
  if (getPartitionIndicesSegments())
    removePartitionIndicesSegmentsAttr();
  if (getPartitionOffsetsSegments())
    removePartitionOffsetsSegmentsAttr();
  if (getPartitionSizesSegments())
    removePartitionSizesSegmentsAttr();
  if (getPartitionEntryModes())
    removePartitionEntryModesAttr();
}

bool DbAcquireOp::hasAllFineGrainedEntries() {
  if (!hasMultiplePartitionEntries())
    return false;
  for (size_t i = 0; i < getNumPartitionEntries(); ++i) {
    if (getPartitionEntryMode(i) != PartitionMode::fine_grained)
      return false;
  }
  return true;
}

/// Helper to add DbAcquireOp operands and attributes to OperationState
static void addDbAcquireOperandsAndAttrs(
    OpBuilder &builder, OperationState &state, ArtsMode mode, Value sourceGuid,
    Value sourcePtr, Type guidType, Type ptrType,
    std::optional<PartitionMode> partitionMode, ArrayRef<Value> indices,
    ArrayRef<Value> offsets, ArrayRef<Value> sizes,
    ArrayRef<Value> partitionIndices, ArrayRef<Value> partitionOffsets,
    ArrayRef<Value> partitionSizes, Value boundsValid,
    ArrayRef<Value> elementOffsets, ArrayRef<Value> elementSizes) {
  state.addTypes({guidType, ptrType});
  state.addAttribute("mode", ArtsModeAttr::get(builder.getContext(), mode));

  if (sourceGuid)
    state.addOperands(sourceGuid);
  state.addOperands(sourcePtr);
  state.addOperands(indices);
  state.addOperands(offsets);
  state.addOperands(sizes);
  state.addOperands(partitionIndices);
  state.addOperands(partitionOffsets);
  state.addOperands(partitionSizes);
  if (boundsValid)
    state.addOperands(boundsValid);
  state.addOperands(elementOffsets);
  state.addOperands(elementSizes);

  state.addAttribute(
      DbAcquireOp::getOperandSegmentSizesAttrName(state.name),
      builder.getDenseI32ArrayAttr(
          {sourceGuid ? 1 : 0, 1, static_cast<int32_t>(indices.size()),
           static_cast<int32_t>(offsets.size()),
           static_cast<int32_t>(sizes.size()),
           static_cast<int32_t>(partitionIndices.size()),
           static_cast<int32_t>(partitionOffsets.size()),
           static_cast<int32_t>(partitionSizes.size()), boundsValid ? 1 : 0,
           static_cast<int32_t>(elementOffsets.size()),
           static_cast<int32_t>(elementSizes.size())}));

  /// Resolve partition_mode from parameter or parent allocation
  PartitionMode resolvedMode = PartitionMode::coarse;
  if (partitionMode.has_value()) {
    resolvedMode = *partitionMode;
  } else if (auto *allocOp = DbUtils::getUnderlyingDbAlloc(sourcePtr)) {
    if (auto parentMode = arts::getPartitionMode(allocOp))
      resolvedMode = *parentMode;
  }
  state.addAttribute(
      AttrNames::Operation::PartitionMode,
      PartitionModeAttr::get(builder.getContext(), resolvedMode));
}

void DbAcquireOp::build(OpBuilder &builder, OperationState &state,
                        ArtsMode mode, Value sourceGuid, Value sourcePtr,
                        std::optional<PartitionMode> partitionMode,
                        SmallVector<Value> indices, SmallVector<Value> offsets,
                        SmallVector<Value> sizes,
                        SmallVector<Value> partitionIndices,
                        SmallVector<Value> partitionOffsets,
                        SmallVector<Value> partitionSizes, Value boundsValid,
                        SmallVector<Value> elementOffsets,
                        SmallVector<Value> elementSizes) {
  auto sourceDb = DbUtils::getUnderlyingDb(sourcePtr);
  assert(sourceDb && "Expected sourceDb");
  auto *rawSourceDbAlloc = DbUtils::getUnderlyingDbAlloc(sourcePtr);
  assert(rawSourceDbAlloc && "Expected sourceDbAlloc");
  auto sourceDbAlloc = dyn_cast<DbAllocOp>(rawSourceDbAlloc);

  auto sourceSizes = DbUtils::getSizesFromDb(sourceDb);
  const uint64_t sourceRank = sourceSizes.size();
  const uint64_t pinnedDims = indices.size();

  if (!indices.empty())
    assert(pinnedDims <= sourceRank &&
           "Number of DB-space indices exceeds source rank");

  uint64_t remainingRank = std::max(sourceRank - pinnedDims, uint64_t(1));

  /// Auto-fill sizes/offsets when indices are provided
  if (!indices.empty()) {
    if (sizes.empty()) {
      sizes.reserve(remainingRank);
      for (uint64_t d = 0; d < remainingRank; ++d)
        sizes.push_back(arts::createOneIndex(builder, state.location));
    }
    if (offsets.empty()) {
      offsets.reserve(remainingRank);
      for (uint64_t d = 0; d < remainingRank; ++d)
        offsets.push_back(arts::createZeroIndex(builder, state.location));
    }
  }

  Type guidType = computeGuidType(builder, sizes);
  Type ptrType = cast<MemRefType>(sourceDbAlloc.getPtr().getType());

  addDbAcquireOperandsAndAttrs(builder, state, mode, sourceGuid, sourcePtr,
                               guidType, ptrType, partitionMode, indices,
                               offsets, sizes, partitionIndices,
                               partitionOffsets, partitionSizes, boundsValid,
                               elementOffsets, elementSizes);
}

void DbAcquireOp::build(
    OpBuilder &builder, OperationState &state, ArtsMode mode, Value sourceGuid,
    Value sourcePtr, Type ptrType, std::optional<PartitionMode> partitionMode,
    SmallVector<Value> indices, SmallVector<Value> offsets,
    SmallVector<Value> sizes, SmallVector<Value> partitionIndices,
    SmallVector<Value> partitionOffsets, SmallVector<Value> partitionSizes,
    Value boundsValid, SmallVector<Value> elementOffsets,
    SmallVector<Value> elementSizes) {
  /// Auto-fill sizes/offsets when indices are provided
  if (!indices.empty()) {
    if (sizes.empty())
      sizes.push_back(arts::createOneIndex(builder, state.location));
    if (offsets.empty()) {
      for (size_t i = 0; i < sizes.size(); ++i)
        offsets.push_back(arts::createZeroIndex(builder, state.location));
    }
  }

  Type guidType = computeGuidType(builder, sizes);

  addDbAcquireOperandsAndAttrs(builder, state, mode, sourceGuid, sourcePtr,
                               guidType, ptrType, partitionMode, indices,
                               offsets, sizes, partitionIndices,
                               partitionOffsets, partitionSizes, boundsValid,
                               elementOffsets, elementSizes);
}

void LoweringContractOp::build(OpBuilder &builder, OperationState &state,
                               Value target, const LoweringContractInfo &info) {
  std::optional<int64_t> contractKind;
  if (info.pattern.kind != ContractKind::Unknown)
    contractKind = static_cast<int64_t>(info.pattern.kind);
  build(builder, state, target, info.pattern.depPattern,
        info.pattern.distributionKind, info.pattern.distributionPattern,
        info.pattern.distributionVersion, info.pattern.revision,
        SmallVector<int64_t>(info.spatial.ownerDims.begin(),
                             info.spatial.ownerDims.end()),
        SmallVector<Value>(info.spatial.blockShape.begin(),
                           info.spatial.blockShape.end()),
        SmallVector<Value>(info.spatial.minOffsets.begin(),
                           info.spatial.minOffsets.end()),
        SmallVector<Value>(info.spatial.maxOffsets.begin(),
                           info.spatial.maxOffsets.end()),
        SmallVector<Value>(info.spatial.writeFootprint.begin(),
                           info.spatial.writeFootprint.end()),
        info.spatial.centerOffset, info.spatial.supportedBlockHalo,
        SmallVector<int64_t>(info.spatial.spatialDims.begin(),
                             info.spatial.spatialDims.end()),
        SmallVector<int64_t>(info.spatial.stencilIndependentDims.begin(),
                             info.spatial.stencilIndependentDims.end()),
        info.analysis.narrowableDep, info.analysis.postDbRefined,
        info.analysis.criticalPathDistance, contractKind);
}

LoweringContractOp
LoweringContractOp::create(OpBuilder &builder, Location location, Value target,
                           const LoweringContractInfo &info) {
  OperationState state(location, getOperationName());
  build(builder, state, target, info);
  auto result = llvm::dyn_cast<LoweringContractOp>(builder.create(state));
  assert(result && "builder didn't return the right type");
  return result;
}

namespace {
static PatternAttr
buildPatternAttr(OpBuilder &builder, std::optional<ArtsDepPattern> depPattern,
                 std::optional<EdtDistributionKind> distributionKind,
                 std::optional<EdtDistributionPattern> distributionPattern,
                 std::optional<int64_t> distributionVersion,
                 std::optional<int64_t> revision) {
  auto *ctx = builder.getContext();
  auto i64Type = IntegerType::get(ctx, 64);
  ArtsDepPatternAttr depAttr = depPattern
                                   ? ArtsDepPatternAttr::get(ctx, *depPattern)
                                   : ArtsDepPatternAttr();
  EdtDistributionKindAttr kindAttr =
      distributionKind ? EdtDistributionKindAttr::get(ctx, *distributionKind)
                       : EdtDistributionKindAttr();
  EdtDistributionPatternAttr patternAttr =
      distributionPattern
          ? EdtDistributionPatternAttr::get(ctx, *distributionPattern)
          : EdtDistributionPatternAttr();
  IntegerAttr versionAttr =
      distributionVersion ? IntegerAttr::get(i64Type, *distributionVersion)
                          : IntegerAttr();
  IntegerAttr revisionAttr =
      revision ? IntegerAttr::get(i64Type, *revision) : IntegerAttr();

  if (!depAttr && !kindAttr && !patternAttr && !versionAttr && !revisionAttr)
    return PatternAttr();
  return PatternAttr::get(ctx, depAttr, kindAttr, patternAttr, versionAttr,
                          revisionAttr);
}

static ContractAttr buildContractAttr(
    OpBuilder &builder, ArrayRef<int64_t> ownerDims,
    std::optional<int64_t> centerOffset, ArrayRef<int64_t> spatialDims,
    ArrayRef<int64_t> stencilIndependentDims, bool supportedBlockHalo,
    bool narrowableDep, bool postDbRefined,
    std::optional<int64_t> criticalPathDistance,
    std::optional<int64_t> contractKind) {
  auto *ctx = builder.getContext();
  auto i64Type = IntegerType::get(ctx, 64);

  DenseI64ArrayAttr ownerDimsAttr =
      ownerDims.empty() ? DenseI64ArrayAttr()
                        : builder.getDenseI64ArrayAttr(ownerDims);
  IntegerAttr centerOffsetAttr =
      centerOffset ? IntegerAttr::get(i64Type, *centerOffset) : IntegerAttr();
  DenseI64ArrayAttr spatialDimsAttr =
      spatialDims.empty() ? DenseI64ArrayAttr()
                          : builder.getDenseI64ArrayAttr(spatialDims);
  DenseI64ArrayAttr stencilIndependentDimsAttr =
      stencilIndependentDims.empty()
          ? DenseI64ArrayAttr()
          : builder.getDenseI64ArrayAttr(stencilIndependentDims);
  BoolAttr supportedBlockHaloAttr =
      supportedBlockHalo ? builder.getBoolAttr(true) : BoolAttr();
  BoolAttr narrowableDepAttr =
      narrowableDep ? builder.getBoolAttr(true) : BoolAttr();
  BoolAttr postDbRefinedAttr =
      postDbRefined ? builder.getBoolAttr(true) : BoolAttr();
  IntegerAttr criticalPathDistanceAttr =
      criticalPathDistance ? IntegerAttr::get(i64Type, *criticalPathDistance)
                           : IntegerAttr();
  IntegerAttr contractKindAttr =
      contractKind ? IntegerAttr::get(i64Type, *contractKind) : IntegerAttr();

  if (!ownerDimsAttr && !centerOffsetAttr && !spatialDimsAttr &&
      !stencilIndependentDimsAttr && !supportedBlockHaloAttr &&
      !narrowableDepAttr && !postDbRefinedAttr && !criticalPathDistanceAttr &&
      !contractKindAttr) {
    return ContractAttr();
  }

  return ContractAttr::get(
      ctx, ownerDimsAttr, centerOffsetAttr, spatialDimsAttr,
      stencilIndependentDimsAttr, supportedBlockHaloAttr, narrowableDepAttr,
      postDbRefinedAttr, criticalPathDistanceAttr, contractKindAttr);
}

static std::optional<int64_t> getOptionalI64(IntegerAttr attr) {
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}
} // namespace

void LoweringContractOp::build(
    OpBuilder &builder, OperationState &state, Value target,
    std::optional<ArtsDepPattern> depPattern,
    std::optional<EdtDistributionKind> distributionKind,
    std::optional<EdtDistributionPattern> distributionPattern,
    std::optional<int64_t> distributionVersion, std::optional<int64_t> revision,
    SmallVector<int64_t> ownerDims, SmallVector<Value> blockShape,
    SmallVector<Value> minOffsets, SmallVector<Value> maxOffsets,
    SmallVector<Value> writeFootprint, std::optional<int64_t> centerOffset,
    bool supportedBlockHalo, SmallVector<int64_t> spatialDims,
    SmallVector<int64_t> stencilIndependentDims, bool narrowableDep,
    bool postDbRefined, std::optional<int64_t> criticalPathDistance,
    std::optional<int64_t> contractKind) {
  state.addOperands(target);
  state.addOperands(blockShape);
  state.addOperands(minOffsets);
  state.addOperands(maxOffsets);
  state.addOperands(writeFootprint);
  state.addAttribute(
      LoweringContractOp::getOperandSegmentSizesAttrName(state.name),
      builder.getDenseI32ArrayAttr(
          {1, static_cast<int32_t>(blockShape.size()),
           static_cast<int32_t>(minOffsets.size()),
           static_cast<int32_t>(maxOffsets.size()),
           static_cast<int32_t>(writeFootprint.size())}));

  if (PatternAttr patternAttr = buildPatternAttr(
          builder, depPattern, distributionKind, distributionPattern,
          distributionVersion, revision)) {
    state.addAttribute("pattern", patternAttr);
  }
  if (ContractAttr contractAttr = buildContractAttr(
          builder, ownerDims, centerOffset, spatialDims, stencilIndependentDims,
          supportedBlockHalo, narrowableDep, postDbRefined,
          criticalPathDistance, contractKind)) {
    state.addAttribute("contract", contractAttr);
  }
}

std::optional<ArtsDepPattern> LoweringContractOp::getDepPattern() {
  if (auto pattern = getPattern())
    if (ArtsDepPatternAttr depPattern = pattern->getDepPattern())
      return depPattern.getValue();
  return std::nullopt;
}

std::optional<EdtDistributionKind> LoweringContractOp::getDistributionKind() {
  if (auto pattern = getPattern())
    if (EdtDistributionKindAttr kind = pattern->getDistributionKind())
      return kind.getValue();
  return std::nullopt;
}

std::optional<EdtDistributionPattern>
LoweringContractOp::getDistributionPattern() {
  if (auto pattern = getPattern())
    if (EdtDistributionPatternAttr distributionPattern =
            pattern->getDistributionPattern()) {
      return distributionPattern.getValue();
    }
  return std::nullopt;
}

std::optional<int64_t> LoweringContractOp::getDistributionVersion() {
  if (auto pattern = getPattern())
    return getOptionalI64(pattern->getDistributionVersion());
  return std::nullopt;
}

std::optional<int64_t> LoweringContractOp::getPatternRevision() {
  if (auto pattern = getPattern())
    return getOptionalI64(pattern->getRevision());
  return std::nullopt;
}

std::optional<SmallVector<int64_t, 4>> LoweringContractOp::getOwnerDims() {
  if (auto contract = getContract())
    if (DenseI64ArrayAttr ownerDims = contract->getOwnerDims())
      return SmallVector<int64_t, 4>(ownerDims.asArrayRef());
  return std::nullopt;
}

std::optional<int64_t> LoweringContractOp::getCenterOffset() {
  if (auto contract = getContract())
    return getOptionalI64(contract->getCenterOffset());
  return std::nullopt;
}

std::optional<SmallVector<int64_t, 4>> LoweringContractOp::getSpatialDims() {
  if (auto contract = getContract())
    if (DenseI64ArrayAttr spatialDims = contract->getSpatialDims())
      return SmallVector<int64_t, 4>(spatialDims.asArrayRef());
  return std::nullopt;
}

std::optional<SmallVector<int64_t, 4>>
LoweringContractOp::getStencilIndependentDims() {
  if (auto contract = getContract())
    if (DenseI64ArrayAttr dims = contract->getStencilIndependentDims())
      return SmallVector<int64_t, 4>(dims.asArrayRef());
  return std::nullopt;
}

std::optional<bool> LoweringContractOp::getSupportedBlockHalo() {
  if (auto contract = getContract())
    if (BoolAttr supported = contract->getSupportedBlockHalo())
      return supported.getValue();
  return std::nullopt;
}

std::optional<bool> LoweringContractOp::getNarrowableDep() {
  if (auto contract = getContract())
    if (BoolAttr narrowable = contract->getNarrowableDep())
      return narrowable.getValue();
  return std::nullopt;
}

std::optional<bool> LoweringContractOp::getPostDbRefined() {
  if (auto contract = getContract())
    if (BoolAttr postDbRefined = contract->getPostDbRefined())
      return postDbRefined.getValue();
  return std::nullopt;
}

std::optional<int64_t> LoweringContractOp::getCriticalPathDistance() {
  if (auto contract = getContract())
    return getOptionalI64(contract->getCriticalPathDistance());
  return std::nullopt;
}

std::optional<int64_t> LoweringContractOp::getContractKind() {
  if (auto contract = getContract())
    return getOptionalI64(contract->getContractKind());
  return std::nullopt;
}

LogicalResult DbAcquireOp::verify() {
  size_t numSizes = getSizes().size();
  size_t numOffsets = getOffsets().size();
  size_t numIndices = getIndices().size();

  /// Validate offsets/sizes consistency
  if (numSizes != 0 && numOffsets != 0 && numOffsets != numSizes) {
    return emitOpError("Number of DB-space offsets (")
           << numOffsets << ") must match Number of DB-space sizes ("
           << numSizes << ")";
  }

  /// Skip validation for partition hints only mode (no indices or sizes)
  if (numIndices == 0 && numSizes == 0)
    return success();

  size_t srcRank = DbUtils::getSizesFromDb(getSourcePtr()).size();

  if (numIndices > srcRank) {
    return emitOpError("Number of DB-space indices (")
           << numIndices << ") exceeds source datablock rank (" << srcRank
           << "). This indicates a dimensionality mismatch between DbAcquire "
           << "and its source DbAlloc.";
  }

  size_t remainingRank = srcRank - numIndices;

  if (remainingRank > 0 && numSizes > 0 && remainingRank != numSizes) {
    return emitOpError("Source rank (")
           << srcRank << ") - indices (" << numIndices
           << ") = " << remainingRank
           << " must equal Number of DB-space sizes (" << numSizes
           << "). When modifying DbAcquire layout, ensure the "
           << "source DbAlloc dimensionality is updated correspondingly.";
  }

  /// Validate indices are not used on single-element datablocks
  if (numIndices > 0) {
    Operation *sourceDb = DbUtils::getUnderlyingDb(getSourcePtr());
    if (DbAnalysis::hasSingleSize(sourceDb)) {
      auto partMode = getPartitionMode();
      bool awaitingPartitioning =
          partMode && (*partMode == PartitionMode::fine_grained ||
                       *partMode == PartitionMode::block);
      if (!awaitingPartitioning) {
        return emitOpError("Cannot use DB-space indices on single-element "
                           "datablock (size=1)\n")
               << *getOperation();
      }
    }
  }

  return success();
}

LogicalResult LoweringContractOp::verify() {
  auto ownerDims = getOwnerDims();
  size_t expectedRank = ownerDims ? ownerDims->size() : 0;
  bool hasOffsetOrFootprintPayload = !getMinOffsets().empty() ||
                                     !getMaxOffsets().empty() ||
                                     !getWriteFootprint().empty();

  if (expectedRank == 0 && hasOffsetOrFootprintPayload)
    return emitOpError(
        "min_offsets/max_offsets/write_footprint require owner_dims");

  if (ownerDims) {
    llvm::DenseSet<int64_t> seenOwnerDims;
    for (int64_t dim : *ownerDims) {
      if (dim < 0)
        return emitOpError("owner_dims must be non-negative");
      if (!seenOwnerDims.insert(dim).second)
        return emitOpError("owner_dims contains duplicate value: ") << dim;
    }
  }

  auto verifyRankedOperands = [&](OperandRange values,
                                  StringRef name) -> LogicalResult {
    if (expectedRank != 0 && !values.empty() && values.size() != expectedRank)
      return emitOpError() << name << " rank (" << values.size()
                           << ") must match owner_dims rank (" << expectedRank
                           << ")";
    return success();
  };

  if (failed(verifyRankedOperands(getBlockShape(), "block_shape")) ||
      failed(verifyRankedOperands(getMinOffsets(), "min_offsets")) ||
      failed(verifyRankedOperands(getMaxOffsets(), "max_offsets")) ||
      failed(verifyRankedOperands(getWriteFootprint(), "write_footprint")))
    return failure();

  if (getMinOffsets().size() != getMaxOffsets().size())
    return emitOpError("min_offsets and max_offsets must have the same rank");

  if (getSupportedBlockHalo().value_or(false)) {
    auto depPattern = getDepPattern();
    if (!depPattern || !isStencilFamilyDepPattern(*depPattern))
      return emitOpError(
          "supported_block_halo requires a stencil-family dep_pattern");
  }

  return success();
}

void DbRefOp::build(OpBuilder &builder, OperationState &state, Value source,
                    ArrayRef<Value> indices) {
  auto *rawDbAlloc = DbUtils::getUnderlyingDbAlloc(source);

  auto sourceMemRefType = dyn_cast<MemRefType>(source.getType());
  if (sourceMemRefType) {
    if (auto sourceElementType =
            dyn_cast<MemRefType>(sourceMemRefType.getElementType())) {
      state.addTypes(sourceElementType);
      state.addOperands(source);
      state.addOperands(indices);
      return;
    }
  }

  // If source is a BlockArgument or otherwise doesn't trace to a DbAllocOp,
  // extract the result type from the source memref's element type.
  if (!rawDbAlloc) {
    assert(sourceMemRefType && "Expected source to be a memref type");
    Type resultType = sourceMemRefType.getElementType();
    state.addTypes(resultType);
    state.addOperands(source);
    state.addOperands(indices);
    return;
  }

  DbAllocOp dbAllocOp = dyn_cast<DbAllocOp>(rawDbAlloc);
  state.addTypes(dbAllocOp.getAllocatedElementType());
  state.addOperands(source);
  state.addOperands(indices);
}

LogicalResult DbRefOp::verify() {
  auto underlyingDbOp = DbUtils::getUnderlyingDb(getSource());
  if (!underlyingDbOp)
    return emitOpError("source must be a datablock operation\n")
           << *getOperation();

  auto outerSizes = DbUtils::getSizesFromDb(underlyingDbOp);
  if (getIndices().size() != outerSizes.size()) {
    return emitOpError("expects ")
           << outerSizes.size() << " index operands (got "
           << getIndices().size() << ")\n"
           << *getOperation();
  }

  SmallVector<Value> allocSizes = outerSizes;
  if (auto dbAcquire = dyn_cast<DbAcquireOp>(underlyingDbOp)) {
    if (auto dbAlloc = dyn_cast_or_null<DbAllocOp>(
            DbUtils::getUnderlyingDbAlloc(dbAcquire.getSourcePtr()))) {
      allocSizes = SmallVector<Value>(dbAlloc.getSizes().begin(),
                                      dbAlloc.getSizes().end());
    }
  }
  bool isCoarse = !allocSizes.empty() && llvm::all_of(allocSizes, [](Value v) {
    int64_t val;
    return ValueAnalysis::getConstantIndex(v, val) && val == 1;
  });
  if (isCoarse) {
    for (Value idx : getIndices()) {
      int64_t val;
      if (!ValueAnalysis::getConstantIndex(idx, val) || val != 0)
        return emitOpError("Coarse-grained datablock expects db_ref indices ")
               << "to be constant zero\n"
               << *getOperation();
    }
  }

  DbAllocOp dbAllocOp = nullptr;
  if (auto dbAcquire = dyn_cast<DbAcquireOp>(underlyingDbOp)) {
    Operation *rawAlloc =
        DbUtils::getUnderlyingDbAlloc(dbAcquire.getSourcePtr());
    if (rawAlloc)
      dbAllocOp = dyn_cast<DbAllocOp>(rawAlloc);
  } else
    dbAllocOp = dyn_cast<DbAllocOp>(underlyingDbOp);

  /// Verify output type
  auto resultType = cast<MemRefType>(getResult().getType());
  auto sourceMemRefType = cast<MemRefType>(getSource().getType());
  if (resultType == sourceMemRefType.getElementType())
    return success();
  if (dbAllocOp && resultType == dbAllocOp.getAllocatedElementType())
    return success();

  return emitOpError("result type must match source element type\n")
         << *getOperation();
}

/// Custom builders for DbNumElementsOp that automatically extract sizes
void DbNumElementsOp::build(OpBuilder &builder, OperationState &state,
                            DbAllocOp allocOp) {
  /// Extract sizes from the DbAllocOp
  SmallVector<Value> sizes = allocOp.getSizes();
  build(builder, state, sizes);
}

void DbNumElementsOp::build(OpBuilder &builder, OperationState &state,
                            DbAcquireOp acquireOp) {
  SmallVector<Value> sizes = acquireOp.getSizes();
  build(builder, state, sizes);
}

struct FoldDbNumElementsSingleSize : public OpRewritePattern<DbNumElementsOp> {
  using OpRewritePattern<DbNumElementsOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(DbNumElementsOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getSizes().size() == 1) {
      rewriter.replaceOp(op, op.getSizes()[0]);
      return success();
    }
    return failure();
  }
};

void DbNumElementsOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.add<FoldDbNumElementsSingleSize>(context);
}

void OmpDepOp::build(OpBuilder &builder, OperationState &state, ArtsMode mode,
                     Value source, SmallVector<Value> indices,
                     SmallVector<Value> sizes) {
  state.addTypes(source.getType());
  state.addAttribute("mode", ArtsModeAttr::get(builder.getContext(), mode));
  state.addOperands(source);
  state.addOperands(indices);
  state.addOperands(sizes);
  state.addAttribute(
      OmpDepOp::getOperandSegmentSizesAttrName(state.name),
      builder.getDenseI32ArrayAttr({1, static_cast<int32_t>(indices.size()),
                                    static_cast<int32_t>(sizes.size())}));
}

void arts::ForOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                              MLIRContext *context) {}

///===----------------------------------------------------------------------===///
/// ForOp LoopLikeOpInterface implementation
///===----------------------------------------------------------------------===///

SmallVector<Region *> arts::ForOp::getLoopRegions() { return {&getRegion()}; }

std::optional<SmallVector<Value>> arts::ForOp::getLoopInductionVars() {
  auto numIVs = getLowerBound().size();
  SmallVector<Value> ivs;
  for (unsigned i = 0; i < numIVs; ++i)
    ivs.push_back(getRegion().getArgument(i));
  return ivs;
}

std::optional<SmallVector<OpFoldResult>> arts::ForOp::getLoopLowerBounds() {
  return SmallVector<OpFoldResult>(getLowerBound().begin(),
                                   getLowerBound().end());
}

std::optional<SmallVector<OpFoldResult>> arts::ForOp::getLoopUpperBounds() {
  return SmallVector<OpFoldResult>(getUpperBound().begin(),
                                   getUpperBound().end());
}

std::optional<SmallVector<OpFoldResult>> arts::ForOp::getLoopSteps() {
  return SmallVector<OpFoldResult>(getStep().begin(), getStep().end());
}
