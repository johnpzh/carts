///==========================================================================///
/// File: RaiseToLinalg.cpp
///
/// Raises supported perfectly nested, affine `sde.su_iterate` bodies to
/// transient `linalg.generic` carriers when the body is structurally
/// compatible. The pass stamps SDE-owned loop classification on the source op
/// so later SDE planning and CODIR materialization preserve the semantic
/// contract without rediscovering loop intent downstream.
///
/// Recognized patterns:
///   - elementwise : all-parallel iterators, identity/permutation maps
///   - matmul      : triple-nested, (m,k)*(k,n)->(m,n), reduction on k
///   - stencil     : all-parallel with constant-offset read maps
///   - reduction   : at least one reduction iterator
///
///==========================================================================///

#include "arts/dialect/sde/Analysis/StructuredOpAnalysis.h"
#include "arts/dialect/sde/Transforms/Passes.h"
namespace mlir::arts {
#define GEN_PASS_DEF_RAISETOLINALG
#include "arts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::arts

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include "arts/utils/ValueAnalysis.h"
#include "arts/utils/Debug.h"
ARTS_DEBUG_SETUP(raise_to_linalg);

using namespace mlir;
using namespace mlir::arts;
using namespace mlir::carts;

namespace {
//===----------------------------------------------------------------------===//
// Linalg raising
//===----------------------------------------------------------------------===//

struct InputOperand {
  Value source; // memref or tensor value
  AffineMap indexingMap;
  SmallVector<Operation *> accessOps; // memref::LoadOp or tensor::ExtractOp
};

struct OutputOperand {
  Value source; // memref or tensor value
  AffineMap indexingMap;
  Type elementType;
};

/// Trace a tensor value through tensor.insert chains to find the original
/// tensor block arg or defining value.
static Value traceToTensorSource(Value v) {
  while (auto insertOp = v.getDefiningOp<tensor::InsertOp>())
    v = insertOp.getDest();
  return v;
}

static std::optional<unsigned> findOperandIndex(ArrayRef<InputOperand> inputs,
                                                Value source,
                                                AffineMap indexingMap) {
  for (auto [idx, input] : llvm::enumerate(inputs)) {
    if (input.source == source && input.indexingMap == indexingMap)
      return idx;
  }
  return std::nullopt;
}

static std::optional<unsigned> findOperandIndex(ArrayRef<OutputOperand> outputs,
                                                Value source,
                                                AffineMap indexingMap) {
  for (auto [idx, output] : llvm::enumerate(outputs)) {
    if (output.source == source && output.indexingMap == indexingMap)
      return idx;
  }
  return std::nullopt;
}

static bool isConstantIndex(Value value, int64_t expected) {
  int64_t actual = 0;
  return arts::ValueAnalysis::getConstantIndex(value, actual) && actual == expected;
}

static bool hasZeroLowerUnitStep(sde::SdeSuIterateOp iterOp,
                                 const sde::LoopNestInfo &nest,
                                 unsigned dim) {
  if (dim < iterOp.getLowerBounds().size() && dim < iterOp.getSteps().size())
    return isConstantIndex(iterOp.getLowerBounds()[dim], 0) &&
           isConstantIndex(iterOp.getSteps()[dim], 1);

  if (dim >= nest.ivs.size())
    return false;
  auto blockArg = dyn_cast<BlockArgument>(nest.ivs[dim]);
  if (!blockArg)
    return false;

  auto forOp = dyn_cast_or_null<scf::ForOp>(
      blockArg.getOwner()->getParentOp());
  return forOp && isConstantIndex(forOp.getLowerBound(), 0) &&
         isConstantIndex(forOp.getStep(), 1);
}

static bool loopIvsMapDirectlyToLinalgIndices(
    sde::SdeSuIterateOp iterOp, const sde::LoopNestInfo &nest) {
  for (unsigned dim = 0, e = nest.ivs.size(); dim < e; ++dim)
    if (!hasZeroLowerUnitStep(iterOp, nest, dim))
      return false;
  return true;
}

static Value cloneScalarValueIntoRegion(Value value, Block &body,
                                        OpBuilder &builder, IRMapping &mapper) {
  if (Value mapped = mapper.lookupOrNull(value))
    return mapped;

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (blockArg.getOwner() == &body)
      return Value();
    return value;
  }

  Operation *defOp = value.getDefiningOp();
  if (!defOp || defOp->getBlock() != &body)
    return value;

  if (isa<memref::LoadOp, tensor::ExtractOp>(defOp))
    return mapper.lookupOrNull(value);

  // tensor.insert is not a scalar op — skip it.
  if (isa<tensor::InsertOp>(defOp))
    return Value();

  for (Value operand : defOp->getOperands()) {
    Value mappedOperand =
        cloneScalarValueIntoRegion(operand, body, builder, mapper);
    if (!mappedOperand)
      return Value();
  }

  builder.clone(*defOp, mapper);
  return mapper.lookupOrNull(value);
}

/// Key for grouping stencil accesses by (base memref, per-dim offset vector).
struct StencilAccessKey {
  Value memref;
  SmallVector<int64_t> offsets; // per-dim constant offset
};

/// Collect loop bounds for each dimension in the nest. Dim 0 comes from the
/// su_iterate; dims 1+ come from nested scf.for ops.
static bool collectLoopBounds(sde::SdeSuIterateOp iterOp,
                              const sde::LoopNestInfo &nest,
                              SmallVectorImpl<Value> &lowerBounds,
                              SmallVectorImpl<Value> &upperBounds) {
  unsigned numDims = nest.ivs.size();
  if (numDims == 0)
    return false;

  // Dim 0: su_iterate bounds (must be exactly 1-D su_iterate for stencils).
  if (iterOp.getLowerBounds().size() != 1 ||
      iterOp.getUpperBounds().size() != 1)
    return false;
  lowerBounds.push_back(iterOp.getLowerBounds().front());
  upperBounds.push_back(iterOp.getUpperBounds().front());

  // Dims 1+: nested scf.for loops. Walk the nest to find them.
  Block *body = &iterOp.getBody().front();
  // Look through cu_region.
  for (Operation &inner : body->without_terminator()) {
    if (auto cuRegion = dyn_cast<sde::SdeCuRegionOp>(inner);
        cuRegion && cuRegion.getKind() == sde::SdeCuKind::parallel) {
      body = &cuRegion.getBody().front();
      break;
    }
  }

  for (unsigned d = 1; d < numDims; ++d) {
    scf::ForOp innerFor;
    for (Operation &op : body->without_terminator()) {
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        innerFor = forOp;
        break;
      }
    }
    if (!innerFor)
      return false;
    lowerBounds.push_back(innerFor.getLowerBound());
    upperBounds.push_back(innerFor.getUpperBound());
    body = innerFor.getBody();
  }
  return true;
}

/// Raise a stencil loop nest to a linalg.generic carrier using shifted
/// tensor.extract_slice views. Each neighbor access becomes a separate input
/// with an extract_slice at the appropriate offset, and all indexing maps
/// are identity. The stencil geometry is encoded in the slice offsets.
static bool raiseStencilToLinalg(sde::SdeSuIterateOp iterOp,
                                 const sde::LoopNestInfo &nest,
                                 ArrayRef<sde::MemrefAccessEntry> reads,
                                 ArrayRef<sde::MemrefAccessEntry> writes,
                                 ArrayRef<utils::IteratorType> iterTypes) {
  Block *computeBody = sde::getSuIterateComputeBlock(iterOp);
  Operation *terminator = computeBody->getTerminator();
  if (!terminator || !isa<sde::SdeYieldOp>(terminator))
    return false;
  auto oldYield = cast<sde::SdeYieldOp>(terminator);

  unsigned numDims = nest.ivs.size();
  if (numDims == 0)
    return false;

  // Collect per-dim loop bounds.
  SmallVector<Value> lowerBounds, upperBounds;
  if (!collectLoopBounds(iterOp, nest, lowerBounds, upperBounds))
    return false;

  Location loc = iterOp.getLoc();
  OpBuilder builder(oldYield);

  // Compute trip count per dimension: tc_d = ub_d - lb_d.
  SmallVector<Value> tripCounts;
  for (unsigned d = 0; d < numDims; ++d)
    tripCounts.push_back(
        arith::SubIOp::create(builder, loc, upperBounds[d], lowerBounds[d]));

  // Decompose each read access into (memref, per-dim-offset) and group them.
  // Each unique (memref, offset_vector) becomes one linalg input.
  struct StencilInput {
    Value memref;
    SmallVector<int64_t> offsets;
    SmallVector<Operation *> accessOps;
  };
  SmallVector<StencilInput> stencilInputs;
  DenseMap<Value, Value> memrefToTensor; // cache mu_memref_to_tensor

  for (const sde::MemrefAccessEntry &read : reads) {
    if (!isa<memref::LoadOp>(read.op))
      return false;

    // Decompose indexing map into per-dim offsets.
    SmallVector<int64_t> offsets;
    for (unsigned r = 0; r < read.indexingMap.getNumResults(); ++r) {
      auto dimOffset = sde::extractDimOffset(read.indexingMap.getResult(r));
      if (!dimOffset || !dimOffset->dim)
        return false;
      offsets.push_back(dimOffset->offset);
    }

    // Find or create matching input group.
    bool found = false;
    for (auto &input : stencilInputs) {
      if (input.memref == read.memref && input.offsets == offsets) {
        input.accessOps.push_back(read.op);
        found = true;
        break;
      }
    }
    if (!found)
      stencilInputs.push_back({read.memref, offsets, {read.op}});
  }

  // Process writes — must all be identity (offset 0 on every dim).
  if (writes.empty())
    return false;
  struct StencilOutput {
    Value memref;
    Type elementType;
    Operation *writeOp;
  };
  SmallVector<StencilOutput> stencilOutputs;
  for (const sde::MemrefAccessEntry &write : writes) {
    if (!isa<memref::StoreOp>(write.op))
      return false;
    auto mrType = dyn_cast<MemRefType>(write.memref.getType());
    if (!mrType)
      return false;
    stencilOutputs.push_back({write.memref, mrType.getElementType(), write.op});
  }

  // Create mu_memref_to_tensor for each unique base memref.
  auto getOrCreateTensor = [&](Value memref) -> Value {
    auto it = memrefToTensor.find(memref);
    if (it != memrefToTensor.end())
      return it->second;
    auto mrType = cast<MemRefType>(memref.getType());
    auto tTy =
        RankedTensorType::get(mrType.getShape(), mrType.getElementType());
    Value tensor =
        sde::SdeMuMemrefToTensorOp::create(builder, loc, tTy, memref);
    memrefToTensor[memref] = tensor;
    return tensor;
  };

  // Build extract_slice for each stencil input view.
  // For access A[iv_d + offset_d], the slice is:
  //   offset = lb_d + offset_d, size = tc_d, stride = 1
  SmallVector<Value> tensorInputs;
  SmallVector<AffineMap> indexingMaps;
  for (const StencilInput &input : stencilInputs) {
    Value baseTensor = getOrCreateTensor(input.memref);
    auto tensorTy = cast<RankedTensorType>(baseTensor.getType());
    unsigned rank = tensorTy.getRank();

    SmallVector<OpFoldResult> offsets(rank);
    SmallVector<OpFoldResult> sizes(rank);
    SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));

    for (unsigned d = 0; d < rank; ++d) {
      if (input.offsets[d] == 0) {
        offsets[d] = lowerBounds[d];
      } else {
        Value offsetVal =
            arith::ConstantIndexOp::create(builder, loc, input.offsets[d]);
        Value sum =
            arith::AddIOp::create(builder, loc, lowerBounds[d], offsetVal);
        offsets[d] = sum;
      }
      sizes[d] = tripCounts[d];
    }

    Value sliced = tensor::ExtractSliceOp::create(builder, loc, baseTensor,
                                                  offsets, sizes, strides);
    tensorInputs.push_back(sliced);
    indexingMaps.push_back(builder.getMultiDimIdentityMap(numDims));
  }

  // Build extract_slice for each output.
  SmallVector<Value> tensorOutputs;
  SmallVector<Type> resultTypes;
  SmallVector<Value> fullOutputTensors;
  for (const StencilOutput &output : stencilOutputs) {
    Value baseTensor = getOrCreateTensor(output.memref);
    auto tensorTy = cast<RankedTensorType>(baseTensor.getType());
    unsigned rank = tensorTy.getRank();

    SmallVector<OpFoldResult> offsets(rank);
    SmallVector<OpFoldResult> sizes(rank);
    SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));

    for (unsigned d = 0; d < rank; ++d) {
      offsets[d] = lowerBounds[d];
      sizes[d] = tripCounts[d];
    }

    Value sliced = tensor::ExtractSliceOp::create(builder, loc, baseTensor,
                                                  offsets, sizes, strides);
    tensorOutputs.push_back(sliced);
    resultTypes.push_back(sliced.getType());
    fullOutputTensors.push_back(baseTensor);
    indexingMaps.push_back(builder.getMultiDimIdentityMap(numDims));
  }

  // Build linalg.generic with all-identity maps.
  bool cloneFailed = false;
  auto generic = linalg::GenericOp::create(
      builder, loc, resultTypes, tensorInputs, tensorOutputs, indexingMaps,
      iterTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        IRMapping mapper;
        // Map each read load result to the corresponding block arg.
        unsigned argIdx = 0;
        for (const StencilInput &input : stencilInputs) {
          for (Operation *accessOp : input.accessOps) {
            auto loadOp = cast<memref::LoadOp>(accessOp);
            mapper.map(loadOp.getResult(), blockArgs[argIdx]);
          }
          ++argIdx;
        }

        // Clone scalar computation and build yield values.
        SmallVector<Value> yielded(stencilOutputs.size());
        SmallVector<bool> hasYield(stencilOutputs.size(), false);
        unsigned outputArgOffset = stencilInputs.size();

        for (auto [outIdx, output] : llvm::enumerate(stencilOutputs)) {
          auto storeOp = cast<memref::StoreOp>(output.writeOp);
          Value valueToWrite = storeOp.getValue();

          Value mappedValue = cloneScalarValueIntoRegion(
              valueToWrite, *nest.innermostBody, nestedBuilder, mapper);
          if (!mappedValue) {
            cloneFailed = true;
            continue;
          }
          yielded[outIdx] = mappedValue;
          hasYield[outIdx] = true;
        }

        for (unsigned i = 0; i < stencilOutputs.size(); ++i) {
          if (!hasYield[i])
            yielded[i] = blockArgs[outputArgOffset + i];
        }

        linalg::YieldOp::create(nestedBuilder, nestedLoc, yielded);
      });

  if (cloneFailed) {
    generic.erase();
    return false;
  }

  // Create insert_slice for each output result.
  for (auto [idx, result] : llvm::enumerate(generic.getResults())) {
    auto tensorTy = cast<RankedTensorType>(fullOutputTensors[idx].getType());
    unsigned rank = tensorTy.getRank();

    SmallVector<OpFoldResult> offsets(rank);
    SmallVector<OpFoldResult> sizes(rank);
    SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));

    for (unsigned d = 0; d < rank; ++d) {
      offsets[d] = lowerBounds[d];
      sizes[d] = tripCounts[d];
    }

    tensor::InsertSliceOp::create(builder, loc, result, fullOutputTensors[idx],
                                  offsets, sizes, strides);
  }

  // Stencil carriers stay dual-representation: the scalar body (scf.for +
  // memref.load/store) coexists with the carrier. Downstream passes (Tiling,
  // IterationSpaceDecomposition) work on the scalar body. The carrier and its
  // support ops (extract_slice, insert_slice, mu_memref_to_tensor) are dead
  // and will be erased by LowerToMemref's eraseDeadCarriers phase.

  ARTS_DEBUG("  raised stencil to linalg.generic with "
             << stencilInputs.size() << " shifted inputs and "
             << stencilOutputs.size() << " outputs (dual-rep)");
  return true;
}

[[maybe_unused]] static bool raiseToLinalg(
    sde::SdeSuIterateOp iterOp, const sde::LoopNestInfo &nest,
    ArrayRef<sde::MemrefAccessEntry> reads,
    ArrayRef<sde::MemrefAccessEntry> writes,
    ArrayRef<utils::IteratorType> iterTypes) {
  Block *computeBody = sde::getSuIterateComputeBlock(iterOp);
  // Find the yield terminator in the compute block (inside cu_region if
  // present, otherwise the su_iterate body directly).
  Operation *terminator = computeBody->getTerminator();
  if (!terminator || !isa<sde::SdeYieldOp>(terminator))
    return false;
  auto oldYield = cast<sde::SdeYieldOp>(terminator);

  Location loc = iterOp.getLoc();

  SmallVector<InputOperand> inputs;
  for (const sde::MemrefAccessEntry &read : reads) {
    // Accept both tensor.extract and memref.load.
    Value source;
    if (auto extractOp = dyn_cast<tensor::ExtractOp>(read.op))
      source = traceToTensorSource(extractOp.getTensor());
    else if (auto loadOp = dyn_cast<memref::LoadOp>(read.op))
      source = read.memref;
    else
      return false;

    auto inputIdx = findOperandIndex(inputs, source, read.indexingMap);
    if (!inputIdx) {
      inputs.push_back({source, read.indexingMap, {}});
      inputIdx = inputs.size() - 1;
    }
    inputs[*inputIdx].accessOps.push_back(read.op);
  }

  SmallVector<OutputOperand> outputs;
  DenseMap<Operation *, unsigned> writeOperandIndex;
  for (const sde::MemrefAccessEntry &write : writes) {
    // Accept both tensor.insert and memref.store.
    Value source;
    Type elementType;
    if (auto insertOp = dyn_cast<tensor::InsertOp>(write.op)) {
      source = traceToTensorSource(insertOp.getDest());
      auto tensorTy = dyn_cast<RankedTensorType>(source.getType());
      if (!tensorTy)
        return false;
      elementType = tensorTy.getElementType();
    } else if (auto storeOp = dyn_cast<memref::StoreOp>(write.op)) {
      source = write.memref;
      auto memrefTy = dyn_cast<MemRefType>(source.getType());
      if (!memrefTy)
        return false;
      elementType = memrefTy.getElementType();
    } else {
      return false;
    }

    if (findOperandIndex(outputs, source, write.indexingMap))
      return false;

    outputs.push_back({source, write.indexingMap, elementType});
    writeOperandIndex[write.op] = outputs.size() - 1;
  }

  if (outputs.empty())
    return false;
  if (!loopIvsMapDirectlyToLinalgIndices(iterOp, nest))
    return false;

  OpBuilder builder(oldYield);

  SmallVector<AffineMap> indexingMaps;
  indexingMaps.reserve(inputs.size() + outputs.size());

  // Build tensor inputs — use tensor sources directly, wrap memrefs via
  // sde.mu_memref_to_tensor (SDE-native boundary op for external data).
  SmallVector<Value> tensorInputs;
  tensorInputs.reserve(inputs.size());
  for (const InputOperand &input : inputs) {
    Value tensorInput;
    if (auto tensorTy = dyn_cast<RankedTensorType>(input.source.getType())) {
      tensorInput = input.source;
    } else {
      auto mrType = cast<MemRefType>(input.source.getType());
      auto tTy =
          RankedTensorType::get(mrType.getShape(), mrType.getElementType());
      tensorInput =
          sde::SdeMuMemrefToTensorOp::create(builder, loc, tTy, input.source);
    }
    tensorInputs.push_back(tensorInput);
    indexingMaps.push_back(input.indexingMap);
  }

  // Build tensor outputs — use tensor sources directly, wrap memrefs via
  // sde.mu_memref_to_tensor (SDE-native boundary op).
  SmallVector<Value> tensorOutputs;
  SmallVector<Type> resultTypes;
  tensorOutputs.reserve(outputs.size());
  for (const auto &output : outputs) {
    Value tensorOutput;
    RankedTensorType tensorTy;
    if (auto tTy = dyn_cast<RankedTensorType>(output.source.getType())) {
      tensorTy = tTy;
      tensorOutput = output.source;
    } else {
      auto mrType = cast<MemRefType>(output.source.getType());
      tensorTy =
          RankedTensorType::get(mrType.getShape(), mrType.getElementType());
      tensorOutput = sde::SdeMuMemrefToTensorOp::create(builder, loc, tensorTy,
                                                        output.source);
    }
    tensorOutputs.push_back(tensorOutput);
    resultTypes.push_back(tensorTy);
    indexingMaps.push_back(output.indexingMap);
  }

  bool cloneFailed = false;
  auto generic = linalg::GenericOp::create(
      builder, loc, resultTypes, tensorInputs, tensorOutputs, indexingMaps,
      iterTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        IRMapping mapper;
        for (auto [dim, iv] : llvm::enumerate(nest.ivs)) {
          auto indexOp = linalg::IndexOp::create(nestedBuilder, nestedLoc, dim);
          mapper.map(iv, indexOp.getResult());
        }
        for (auto [inputIdx, input] : llvm::enumerate(inputs)) {
          for (Operation *accessOp : input.accessOps) {
            if (auto loadOp = dyn_cast<memref::LoadOp>(accessOp))
              mapper.map(loadOp.getResult(), blockArgs[inputIdx]);
            else if (auto extractOp = dyn_cast<tensor::ExtractOp>(accessOp))
              mapper.map(extractOp.getResult(), blockArgs[inputIdx]);
          }
        }

        SmallVector<Value> yielded(outputs.size());
        SmallVector<bool> hasYield(outputs.size(), false);
        unsigned outputArgOffset = inputs.size();

        for (const sde::MemrefAccessEntry &write : writes) {
          Value valueToWrite;
          if (auto storeOp = dyn_cast<memref::StoreOp>(write.op))
            valueToWrite = storeOp.getValue();
          else if (auto insertOp = dyn_cast<tensor::InsertOp>(write.op))
            valueToWrite = insertOp.getScalar();
          else
            continue;

          auto writeIt = writeOperandIndex.find(write.op);
          if (writeIt == writeOperandIndex.end())
            continue;

          Value mappedValue = cloneScalarValueIntoRegion(
              valueToWrite, *nest.innermostBody, nestedBuilder, mapper);
          if (!mappedValue) {
            cloneFailed = true;
            continue;
          }
          yielded[writeIt->second] = mappedValue;
          hasYield[writeIt->second] = true;
        }

        for (auto [outputIdx, output] : llvm::enumerate(outputs)) {
          if (hasYield[outputIdx])
            continue;
          yielded[outputIdx] = blockArgs[outputArgOffset + outputIdx];
        }

        linalg::YieldOp::create(nestedBuilder, nestedLoc, yielded);
      });

  if (cloneFailed) {
    generic.erase();
    return false;
  }

  // Erase scalar body ops that the carrier now represents. The carrier is
  // authoritative — the scalar ops (memref.load, memref.store, tensor.extract,
  // tensor.insert, and now-dead arith ops) are redundant.
  //
  // Iteratively erase: first drop uses from stores/inserts (side-effecting
  // ops that produce no results), then erase dead ops in reverse order.
  unsigned erased = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> toErase;
    for (Operation &op : llvm::reverse(computeBody->without_terminator())) {
      // Keep the carrier and its support ops.
      if (isa<linalg::GenericOp>(op) || isa<sde::SdeMuMemrefToTensorOp>(op) ||
          isa<bufferization::ToTensorOp>(op) || isa<tensor::EmptyOp>(op))
        continue;
      // Erase stores/inserts (side effects but no SSA results to worry about).
      if (isa<memref::StoreOp, tensor::InsertOp>(&op)) {
        toErase.push_back(&op);
        continue;
      }
      // Erase loads/extracts and pure ops that are now dead.
      if (op.use_empty()) {
        toErase.push_back(&op);
        continue;
      }
    }
    for (Operation *op : toErase) {
      op->erase();
      ++erased;
      changed = true;
    }
  }

  ARTS_DEBUG("  raised sde.su_iterate body to linalg.generic with "
             << inputs.size() << " inputs and " << outputs.size()
             << " outputs (carrier-authoritative, " << erased
             << " scalar ops erased)");
  return true;
}

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct RaiseToLinalgPass
    : public arts::impl::RaiseToLinalgBase<RaiseToLinalgPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    unsigned analyzed = 0;
    unsigned classified = 0;
    unsigned raised = 0;

    module.walk([&](sde::SdeSuIterateOp iterOp) {
      ++analyzed;

      std::optional<sde::StructuredLoopSummary> summary =
          sde::analyzeStructuredLoop(iterOp);
      if (!summary)
        return;

      sde::SdeStructuredClassification pattern = summary->classification;

      iterOp.setStructuredClassificationAttr(
          sde::SdeStructuredClassificationAttr::get(ctx, pattern));
      ++classified;
      if (summary->supportsLinalgCarrier()) {
        bool didRaise = false;
        if (pattern == sde::SdeStructuredClassification::stencil) {
          didRaise = raiseStencilToLinalg(iterOp, summary->nest, summary->reads,
                                          summary->writes, summary->iterTypes);
        } else {
          /// Non-stencil carriers currently materialize full tensor domains.
          /// Inside su_iterate that duplicates work for each distributed IV
          /// chunk. Keep the scalar SDE body authoritative until this path
          /// emits explicit per-iteration tensor slices like the stencil
          /// dual-representation path.
          ARTS_DEBUG("skipping non-slice-aware linalg carrier for '"
                     << stringifySdeStructuredClassification(pattern) << "'");
        }
        if (didRaise)
          ++raised;
      }

      ARTS_DEBUG("classified sde.su_iterate as '"
                 << stringifySdeStructuredClassification(pattern) << "' with "
                 << summary->nest.ivs.size() << " dims ("
                 << summary->reads.size() << " ins, "
                 << summary->outputMaps.size() << " outs)");
    });

    ARTS_INFO("RaiseToLinalg: analyzed " << analyzed << " loops, classified "
                                         << classified << ", raised "
                                         << raised);
  }
};

} // namespace

namespace mlir::carts::sde {

std::unique_ptr<Pass> createRaiseToLinalgPass() {
  return std::make_unique<RaiseToLinalgPass>();
}

} // namespace mlir::carts::sde
