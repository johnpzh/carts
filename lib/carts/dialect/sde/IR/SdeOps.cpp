///==========================================================================///
/// File: SdeOps.cpp
/// Defines SDE dialect operation helpers and verifiers.
///==========================================================================///

#include "carts/dialect/sde/IR/SdeDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::sde;

#define GET_OP_CLASSES
#include "carts/dialect/sde/IR/SdeOps.cpp.inc"

static bool isSdeCpsPortableCarryType(Type type) {
  return type.isIntOrIndexOrFloat() ||
         isa<DepType, CompletionType, TokenType>(type);
}

static bool isSdeCpsBareDataOrPointerType(Type type) {
  return isa<MemRefType, LLVM::LLVMPointerType>(type);
}

static LogicalResult checkSdeCpsPortableCarry(Operation *op, StringRef role,
                                              unsigned index, Type type) {
  if (isSdeCpsPortableCarryType(type))
    return success();

  InFlightDiagnostic diag = op->emitError() << "sde.cps " << role << " #"
                                            << index << " cannot carry ";
  if (isSdeCpsBareDataOrPointerType(type))
    diag << "bare data/pointer type ";
  diag << type << "; use sde.mu_token, sde.mu_dep, or sde.control_token";
  return failure();
}

//===----------------------------------------------------------------------===//
// SdeCuRegionOp — custom assembly format + verifier
//===----------------------------------------------------------------------===//

// Print: sde.cu_region <kind> [nowait]
//        [iter_args(%a = %init : type) -> (type)]
//        { body } [attr-dict]
void SdeCuRegionOp::print(OpAsmPrinter &p) {
  // Print kind enum in angle-bracket form: <parallel>, <single>, <task>
  p << " <" << stringifySdeCuKind(getKind()) << ">";
  if (getNowait())
    p << " nowait";
  if (!getIterArgs().empty()) {
    p << " iter_args(";
    Block &body = getBody().front();
    llvm::interleaveComma(
        llvm::zip(body.getArguments(), getIterArgs()), p, [&](auto pair) {
          p << std::get<0>(pair) << " = " << std::get<1>(pair);
        });
    p << " : ";
    llvm::interleaveComma(getIterArgs().getTypes(), p);
    p << ")";
    p << " -> (";
    llvm::interleaveComma(getResultTypes(), p);
    p << ")";
  }
  p << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/!getIterArgs().empty());
  p.printOptionalAttrDict((*this)->getAttrs(), {"kind", "nowait"});
}

// Parse: sde.cu_region <kind> [nowait]
//        [iter_args(%a = %init : type) -> (type)]
//        { body } [attr-dict]
ParseResult SdeCuRegionOp::parse(OpAsmParser &parser, OperationState &result) {
  MLIRContext *ctx = parser.getContext();

  // Parse kind: <parallel> | <single> | <task>
  {
    SdeCuKindAttr kindAttr;
    if (parser.parseCustomAttributeWithFallback(kindAttr, Type{}, "kind",
                                                result.attributes))
      return failure();
  }

  // Parse optional nowait
  if (succeeded(parser.parseOptionalKeyword("nowait")))
    result.addAttribute("nowait", UnitAttr::get(ctx));

  // Parse optional iter_args
  SmallVector<OpAsmParser::UnresolvedOperand> iterArgOperands;
  SmallVector<OpAsmParser::Argument> bodyArgs;
  SmallVector<Type> iterArgTypes;
  SmallVector<Type> resultTypes;
  bool hasIterArgs = false;

  if (succeeded(parser.parseOptionalKeyword("iter_args"))) {
    hasIterArgs = true;
    if (parser.parseLParen())
      return failure();

    if (failed(parser.parseOptionalRParen())) {
      do {
        OpAsmParser::Argument bodyArg;
        OpAsmParser::UnresolvedOperand initVal;
        if (parser.parseArgument(bodyArg) || parser.parseEqual() ||
            parser.parseOperand(initVal))
          return failure();
        bodyArgs.push_back(bodyArg);
        iterArgOperands.push_back(initVal);
      } while (succeeded(parser.parseOptionalComma()));

      if (parser.parseColon() || parser.parseTypeList(iterArgTypes) ||
          parser.parseRParen())
        return failure();

      if (iterArgTypes.size() != iterArgOperands.size())
        return parser.emitError(parser.getCurrentLocation(),
                                "iter_args type count mismatch");

      for (auto [arg, ty] : llvm::zip(bodyArgs, iterArgTypes))
        arg.type = ty;

      if (parser.resolveOperands(iterArgOperands, iterArgTypes,
                                 parser.getCurrentLocation(), result.operands))
        return failure();

      if (parser.parseArrow() || parser.parseLParen() ||
          parser.parseTypeList(resultTypes) || parser.parseRParen())
        return failure();

      result.addTypes(resultTypes);
    }
  }

  // Parse region
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, bodyArgs,
                         /*enableNameShadowing=*/false))
    return failure();

  // Ensure block exists
  if (body->empty())
    body->push_back(new Block());

  // Match the custom assembly form that omits empty yields for no-result
  // regions.
  Block &entry = body->front();
  if (!hasIterArgs &&
      (entry.empty() || !entry.back().hasTrait<OpTrait::IsTerminator>())) {
    auto endBuilder = OpBuilder::atBlockEnd(&entry);
    SdeYieldOp::create(endBuilder, result.location, ValueRange{});
  }

  // Parse optional attr-dict
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

LogicalResult SdeCuRegionOp::verify() {
  if (getBody().empty())
    return emitOpError() << "expects body to contain a single block";

  Block &entry = getBody().front();
  unsigned numIterArgs = getIterArgs().size();

  if (numIterArgs > 0) {
    // With iter_args: block args must match iter_args types
    if (entry.getNumArguments() != numIterArgs)
      return emitOpError() << "expects " << numIterArgs
                           << " block argument(s) for iter_args; got "
                           << entry.getNumArguments();
    for (auto [i, pair] :
         llvm::enumerate(llvm::zip(entry.getArguments(), getIterArgs()))) {
      auto [blockArg, iterArg] = pair;
      if (blockArg.getType() != iterArg.getType())
        return emitOpError()
               << "block argument #" << i << " type (" << blockArg.getType()
               << ") does not match iter_arg type (" << iterArg.getType()
               << ")";
    }

    // Result count must match iter_args count
    if (getNumResults() != numIterArgs)
      return emitOpError() << "expects " << numIterArgs
                           << " result(s) matching iter_args; got "
                           << getNumResults();

    // Yield must match results
    auto yield = dyn_cast_or_null<SdeYieldOp>(entry.getTerminator());
    if (!yield)
      return emitOpError()
             << "expects body to terminate with sde.yield when iter_args "
                "are present";
    if (yield.getValues().size() != getNumResults())
      return emitOpError() << "sde.yield operand count ("
                           << yield.getValues().size()
                           << ") does not match result count ("
                           << getNumResults() << ")";
    for (auto [i, pair] :
         llvm::enumerate(llvm::zip(yield.getValues(), getResultTypes()))) {
      auto [yielded, resultTy] = pair;
      if (yielded.getType() != resultTy)
        return emitOpError()
               << "sde.yield operand #" << i << " type (" << yielded.getType()
               << ") does not match result type (" << resultTy << ")";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SdeSuIterateOp — custom assembly format + verifier
//===----------------------------------------------------------------------===//

// Assembly format (preserves the shape of the previous declarative format,
// adding optional `iter_args` / result types and always emitting sde.yield):
//
//   sde.su_iterate (%lb) to (%ub) step (%step)
//     [schedule(<kind> [, %chunk])]
//     [nowait]
//     [reduction [<kinds>] (%accs : types)]
//     [reduction_strategy(<strategy>)]
//     [classification(<class>)]
//     [iter_args(%a = %init : type) -> (type)]
//     { body }
//     [attr-dict]

void SdeSuIterateOp::print(OpAsmPrinter &p) {
  // (lb) to (ub) step (step)
  p << " (" << getLowerBounds() << ") to (" << getUpperBounds() << ") step ("
    << getSteps() << ")";

  // schedule(<kind>[, %chunk])
  if (auto sched = getSchedule()) {
    p << " schedule(<" << stringifySdeScheduleKind(*sched) << ">";
    if (getChunkSize())
      p << ", " << getChunkSize();
    p << ")";
  }

  // nowait
  if (getNowait())
    p << " nowait";

  // reduction[<kinds>](%accs : types)
  if (auto reductionKinds = getReductionKindsAttr()) {
    p << " reduction[";
    llvm::interleaveComma(reductionKinds, p);
    p << "](";
    llvm::interleaveComma(getReductionAccumulators(), p,
                          [&](Value v) { p << v; });
    p << " : ";
    llvm::interleaveComma(getReductionAccumulators().getTypes(), p);
    p << ")";
  }

  // reduction_strategy(<strategy>)
  if (auto strategy = getReductionStrategy()) {
    p << " reduction_strategy(<" << stringifySdeReductionStrategy(*strategy)
      << ">)";
  }

  // classification(<class>)
  if (auto cls = getStructuredClassification()) {
    p << " classification(<" << stringifySdeStructuredClassification(*cls)
      << ">)";
  }

  // iter_args(%a = %init : type) -> (type)
  // Block arguments beyond the induction variables carry iter_args.
  unsigned numIVs = getLowerBounds().size();
  unsigned numResults = getNumResults();
  if (numResults > 0) {
    Block &body = getBody().front();
    p << " iter_args(";
    for (unsigned i = 0; i < numResults; ++i) {
      if (i > 0)
        p << ", ";
      p << body.getArgument(numIVs + i) << " = "
        << getReductionAccumulators()[i];
    }
    p << " : ";
    llvm::interleaveComma(getResultTypes(), p);
    p << ") -> (";
    llvm::interleaveComma(getResultTypes(), p);
    p << ")";
  }

  // body — hide entry block args; show yield only when results exist
  p << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/numResults > 0);

  // attr-dict — elide attributes with dedicated syntax
  SmallVector<StringRef> elidedAttrs = {
      "schedule",
      "nowait",
      "reductionKinds",
      "reductionStrategy",
      "structuredClassification",
      getOperandSegmentSizesAttrName().getValue()};
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

ParseResult SdeSuIterateOp::parse(OpAsmParser &parser, OperationState &result) {
  MLIRContext *ctx = parser.getContext();
  auto indexType = IndexType::get(ctx);

  // ---- (lb) to (ub) step (step) ----
  SmallVector<OpAsmParser::UnresolvedOperand> lbOps, ubOps, stepOps;
  if (parser.parseLParen() || parser.parseOperandList(lbOps) ||
      parser.parseRParen() || parser.parseKeyword("to") ||
      parser.parseLParen() || parser.parseOperandList(ubOps) ||
      parser.parseRParen() || parser.parseKeyword("step") ||
      parser.parseLParen() || parser.parseOperandList(stepOps) ||
      parser.parseRParen())
    return failure();

  unsigned numDims = lbOps.size();
  SmallVector<Type> indexTypes(numDims, indexType);
  if (parser.resolveOperands(lbOps, indexTypes, parser.getCurrentLocation(),
                             result.operands) ||
      parser.resolveOperands(ubOps, indexTypes, parser.getCurrentLocation(),
                             result.operands) ||
      parser.resolveOperands(stepOps, indexTypes, parser.getCurrentLocation(),
                             result.operands))
    return failure();

  // ---- optional schedule(<kind>[, %chunk]) ----
  bool hasChunkSize = false;
  if (succeeded(parser.parseOptionalKeyword("schedule"))) {
    SdeScheduleKindAttr schedAttr;
    if (parser.parseLParen() ||
        parser.parseCustomAttributeWithFallback(schedAttr, Type{}, "schedule",
                                                result.attributes))
      return failure();
    if (succeeded(parser.parseOptionalComma())) {
      OpAsmParser::UnresolvedOperand chunkOp;
      if (parser.parseOperand(chunkOp) ||
          parser.resolveOperand(chunkOp, indexType, result.operands))
        return failure();
      hasChunkSize = true;
    }
    if (parser.parseRParen())
      return failure();
  }

  // ---- optional nowait ----
  if (succeeded(parser.parseOptionalKeyword("nowait")))
    result.addAttribute("nowait", UnitAttr::get(ctx));

  // ---- optional reduction[<kinds>](%accs : types) ----
  SmallVector<OpAsmParser::UnresolvedOperand> redAccOps;
  SmallVector<Type> redAccTypes;
  if (succeeded(parser.parseOptionalKeyword("reduction"))) {
    Attribute kindsAttr;
    if (parser.parseLSquare())
      return failure();
    // Parse the array attribute (e.g. [#sde<reduction_kind<add>>])
    if (parser.parseAttribute(kindsAttr))
      return failure();
    if (parser.parseRSquare())
      return failure();
    result.addAttribute("reductionKinds", kindsAttr);

    if (parser.parseLParen() || parser.parseOperandList(redAccOps) ||
        parser.parseColon() || parser.parseTypeList(redAccTypes) ||
        parser.parseRParen() ||
        parser.resolveOperands(redAccOps, redAccTypes,
                               parser.getCurrentLocation(), result.operands))
      return failure();
  }

  // ---- optional reduction_strategy(<strategy>) ----
  if (succeeded(parser.parseOptionalKeyword("reduction_strategy"))) {
    SdeReductionStrategyAttr stratAttr;
    if (parser.parseLParen() ||
        parser.parseCustomAttributeWithFallback(
            stratAttr, Type{}, "reductionStrategy", result.attributes) ||
        parser.parseRParen())
      return failure();
  }

  // ---- optional classification(<class>) ----
  if (succeeded(parser.parseOptionalKeyword("classification"))) {
    SdeStructuredClassificationAttr classAttr;
    if (parser.parseLParen() ||
        parser.parseCustomAttributeWithFallback(
            classAttr, Type{}, "structuredClassification", result.attributes) ||
        parser.parseRParen())
      return failure();
  }

  // ---- optional iter_args(%a = %init : type) -> (type) ----
  SmallVector<OpAsmParser::Argument> iterBodyArgs;
  SmallVector<OpAsmParser::UnresolvedOperand> iterArgOperands;
  SmallVector<Type> iterArgTypes;

  if (succeeded(parser.parseOptionalKeyword("iter_args"))) {
    if (parser.parseLParen())
      return failure();

    if (failed(parser.parseOptionalRParen())) {
      do {
        OpAsmParser::Argument bodyArg;
        OpAsmParser::UnresolvedOperand initVal;
        if (parser.parseArgument(bodyArg) || parser.parseEqual() ||
            parser.parseOperand(initVal))
          return failure();
        iterBodyArgs.push_back(bodyArg);
        iterArgOperands.push_back(initVal);
      } while (succeeded(parser.parseOptionalComma()));

      if (parser.parseColon() || parser.parseTypeList(iterArgTypes) ||
          parser.parseRParen())
        return failure();

      if (iterArgTypes.size() != iterArgOperands.size())
        return parser.emitError(parser.getCurrentLocation(),
                                "iter_args type count mismatch");

      for (auto [arg, ty] : llvm::zip(iterBodyArgs, iterArgTypes))
        arg.type = ty;

      // iter_args init values go into the reductionAccumulators segment.
      if (parser.resolveOperands(iterArgOperands, iterArgTypes,
                                 parser.getCurrentLocation(), result.operands))
        return failure();

      SmallVector<Type> resultTypes;
      if (parser.parseArrow() || parser.parseLParen() ||
          parser.parseTypeList(resultTypes) || parser.parseRParen())
        return failure();
      result.addTypes(resultTypes);
    }
  }

  // ---- Operand segment sizes ----
  // Order: lowerBounds | upperBounds | steps | chunkSize? | redAccs+iterArgs
  SmallVector<int32_t> segmentSizes = {
      static_cast<int32_t>(numDims), static_cast<int32_t>(numDims),
      static_cast<int32_t>(numDims), hasChunkSize ? 1 : 0,
      static_cast<int32_t>(redAccOps.size() + iterArgOperands.size())};
  result.addAttribute(SdeSuIterateOp::getOperandSegmentSizeAttr(),
                      parser.getBuilder().getDenseI32ArrayAttr(segmentSizes));

  // ---- Body region ----
  // Build block arguments: numDims IV args + iter_args body args.
  SmallVector<OpAsmParser::Argument> allBodyArgs;
  for (unsigned i = 0; i < numDims; ++i) {
    OpAsmParser::Argument ivArg;
    ivArg.type = indexType;
    allBodyArgs.push_back(ivArg);
  }
  for (auto &arg : iterBodyArgs)
    allBodyArgs.push_back(arg);

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, allBodyArgs,
                         /*enableNameShadowing=*/false))
    return failure();

  if (body->empty())
    body->push_back(new Block());

  // Auto-insert empty sde.yield when body has no terminator.
  Block &entry = body->front();
  if (entry.empty() || !entry.back().hasTrait<OpTrait::IsTerminator>()) {
    auto endBuilder = OpBuilder::atBlockEnd(&entry);
    SdeYieldOp::create(endBuilder, result.location, ValueRange{});
  }

  // ---- attr-dict ----
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

LogicalResult SdeSuIterateOp::verify() {
  if (getBody().empty())
    return emitOpError() << "expects body to contain a single block";

  Block &entry = getBody().front();

  // Body must have a terminator (sde.yield)
  auto yield = dyn_cast_or_null<SdeYieldOp>(entry.getTerminator());
  if (!yield)
    return emitOpError() << "expects body to terminate with sde.yield";

  // Result count must match yield operand count
  if (yield.getValues().size() != getNumResults())
    return emitOpError() << "sde.yield operand count ("
                         << yield.getValues().size()
                         << ") does not match result count (" << getNumResults()
                         << ")";

  // When results are present, check type consistency
  for (auto [i, pair] :
       llvm::enumerate(llvm::zip(yield.getValues(), getResultTypes()))) {
    auto [yielded, resultTy] = pair;
    if (yielded.getType() != resultTy)
      return emitOpError() << "sde.yield operand #" << i << " type ("
                           << yielded.getType()
                           << ") does not match result type (" << resultTy
                           << ")";
  }

  bool hasCpsGroup = static_cast<bool>(getCpsGroupIdAttr());
  bool hasCpsStageIndex = static_cast<bool>(getCpsStageIndexAttr());
  bool hasCpsStageCount = static_cast<bool>(getCpsStageCountAttr());
  if (hasCpsGroup || hasCpsStageIndex || hasCpsStageCount) {
    if (!hasCpsGroup || !hasCpsStageIndex || !hasCpsStageCount)
      return emitOpError()
             << "sde.cps stage plan requires sde.cps_group_id, "
                "sde.cps_stage_index, and sde.cps_stage_count together";
    if (auto strategy = getAsyncStrategy();
        !strategy || *strategy != SdeAsyncStrategy::cps_chain)
      return emitOpError()
             << "sde.cps stage plan requires sde.async_strategy cps_chain";
    if (auto repetition = getRepetitionStructure();
        !repetition || *repetition != SdeRepetitionStructure::full_timestep)
      return emitOpError()
             << "sde.cps stage plan requires sde.repetition_structure "
                "full_timestep";

    int64_t groupId = getCpsGroupIdAttr().getInt();
    int64_t stageIndex = getCpsStageIndexAttr().getInt();
    int64_t stageCount = getCpsStageCountAttr().getInt();
    if (groupId < 0)
      return emitOpError() << "sde.cps_group_id must be non-negative";
    if (stageCount <= 0)
      return emitOpError() << "sde.cps_stage_count must be positive";
    if (stageIndex < 0 || stageIndex >= stageCount)
      return emitOpError()
             << "sde.cps_stage_index must be in [0, sde.cps_stage_count)";

    for (auto [index, value] : llvm::enumerate(getReductionAccumulators())) {
      if (failed(checkSdeCpsPortableCarry(getOperation(), "carry operand",
                                          index, value.getType())))
        return failure();
    }
    for (auto [index, result] : llvm::enumerate(getResults())) {
      if (failed(checkSdeCpsPortableCarry(getOperation(), "result", index,
                                          result.getType())))
        return failure();
    }
    unsigned numIVs = getLowerBounds().size();
    for (unsigned index = numIVs; index < entry.getNumArguments(); ++index) {
      if (failed(checkSdeCpsPortableCarry(getOperation(), "body argument",
                                          index - numIVs,
                                          entry.getArgument(index).getType())))
        return failure();
    }
  }

  if (auto strategy = getAsyncStrategy();
      strategy && *strategy == SdeAsyncStrategy::cps_chain && !hasCpsGroup)
    return emitOpError()
           << "sde.async_strategy cps_chain requires a sde.cps stage plan";

  return success();
}

SmallVector<Region *> SdeSuIterateOp::getLoopRegions() { return {&getBody()}; }

std::optional<SmallVector<Value>> SdeSuIterateOp::getLoopInductionVars() {
  auto numIVs = getLowerBounds().size();
  if (getBody().getNumArguments() < numIVs)
    return std::nullopt;
  SmallVector<Value> ivs;
  for (unsigned i = 0; i < numIVs; ++i)
    ivs.push_back(getBody().getArgument(i));
  return ivs;
}

std::optional<SmallVector<OpFoldResult>> SdeSuIterateOp::getLoopLowerBounds() {
  return SmallVector<OpFoldResult>(getLowerBounds().begin(),
                                   getLowerBounds().end());
}

std::optional<SmallVector<OpFoldResult>> SdeSuIterateOp::getLoopUpperBounds() {
  return SmallVector<OpFoldResult>(getUpperBounds().begin(),
                                   getUpperBounds().end());
}

std::optional<SmallVector<OpFoldResult>> SdeSuIterateOp::getLoopSteps() {
  return SmallVector<OpFoldResult>(getSteps().begin(), getSteps().end());
}

///===----------------------------------------------------------------------===///
/// SdeMuAllocOp verifier — dynamic dim count must match `?` in result memref.
///===----------------------------------------------------------------------===///
LogicalResult SdeMuAllocOp::verify() {
  auto memrefTy = cast<MemRefType>(getMemref().getType());
  int64_t numDynamic = memrefTy.getNumDynamicDims();
  if (static_cast<int64_t>(getDynamicSizes().size()) != numDynamic)
    return emitOpError() << "expects " << numDynamic
                         << " dynamic size(s) for result type " << memrefTy
                         << "; got " << getDynamicSizes().size();
  return success();
}

///===----------------------------------------------------------------------===///
/// SdeMuTokenOp verifier.
///===----------------------------------------------------------------------===///
LogicalResult SdeMuTokenOp::verify() {
  auto sourceTy = cast<MemRefType>(getSource().getType());
  auto tokenTy = cast<TokenType>(getToken().getType());
  MemRefType sliceTy = tokenTy.getSliceType();

  if (sourceTy.getElementType() != sliceTy.getElementType()) {
    return emitOpError()
           << "expects token slice_type element type to match source type ("
           << sourceTy << " vs " << sliceTy << ")";
  }

  auto offsets = getOffsets();
  auto sizes = getSizes();

  // Offsets and sizes must either both be empty (whole-storage token) or both
  // be supplied.
  if (offsets.size() != sizes.size()) {
    return emitOpError()
           << "expects offsets and sizes to have the same count (got "
           << offsets.size() << " offsets, " << sizes.size() << " sizes)";
  }

  int64_t rank = sourceTy.getRank();
  if (sliceTy.getRank() != rank) {
    return emitOpError() << "expects token slice_type rank ("
                         << sliceTy.getRank() << ") to match source rank ("
                         << rank << ")";
  }

  if (offsets.empty()) {
    if (sliceTy != sourceTy) {
      return emitOpError()
             << "expects whole-storage token slice_type to match source type ("
             << sourceTy << "), got " << sliceTy;
    }
    return success();
  }

  // Rank match between offsets/sizes and source storage.
  if (static_cast<int64_t>(offsets.size()) != rank) {
    return emitOpError() << "expects offsets/sizes count (" << offsets.size()
                         << ") to match source rank (" << rank << ")";
  }

  // Static sizes must be non-negative; when both offset and size are constants
  // for a dimension, offset + size <= source_dim.
  ArrayRef<int64_t> sourceShape = sourceTy.getShape();
  ArrayRef<int64_t> sliceShape = sliceTy.getShape();
  for (int64_t i = 0; i < rank; ++i) {
    APInt sizeVal;
    bool sizeIsConst = matchPattern(sizes[i], m_ConstantInt(&sizeVal));
    if (sizeIsConst && sizeVal.isNegative()) {
      return emitOpError() << "expects non-negative size at dimension " << i
                           << " (got " << sizeVal.getSExtValue() << ")";
    }

    APInt offsetVal;
    bool offsetIsConst = matchPattern(offsets[i], m_ConstantInt(&offsetVal));

    if (sizeIsConst && !ShapedType::isDynamic(sliceShape[i]) &&
        sliceShape[i] != sizeVal.getSExtValue()) {
      return emitOpError() << "expects token slice_type dimension " << i << " ("
                           << sliceShape[i] << ") to match static token size "
                           << sizeVal.getSExtValue();
    }

    if (sizeIsConst && offsetIsConst &&
        !ShapedType::isDynamic(sourceShape[i])) {
      int64_t off = offsetVal.getSExtValue();
      int64_t sz = sizeVal.getSExtValue();
      if (off < 0) {
        return emitOpError() << "expects non-negative offset at dimension " << i
                             << " (got " << off << ")";
      }
      if (off + sz > sourceShape[i]) {
        return emitOpError()
               << "slice at dimension " << i << " (offset=" << off
               << ", size=" << sz
               << ") exceeds source bound (dim=" << sourceShape[i] << ")";
      }
    }
  }

  return success();
}

///===----------------------------------------------------------------------===///
/// SdeCuCodeletOp verifier.
///===----------------------------------------------------------------------===///
LogicalResult SdeCuCodeletOp::verify() {
  auto tokens = getTokens();
  auto captures = getCaptures();

  auto isScalarCaptureType = [](Type type) {
    return type.isIntOrIndexOrFloat();
  };

  // Every token operand must be `!sde.token<memref<...>>`.
  SmallVector<TokenType> tokenTypes;
  tokenTypes.reserve(tokens.size());
  for (auto token : tokens) {
    auto tt = llvm::dyn_cast<TokenType>(token.getType());
    if (!tt)
      return emitOpError() << "expects every operand to be of type !sde.token";
    tokenTypes.push_back(tt);
  }

  // Body has exactly one block.
  if (getBody().empty())
    return emitOpError() << "expects body to contain a single block";
  Block &entry = getBody().front();

  // Block arguments first mirror token slice types, then scalar captures.
  unsigned expectedArgs = tokens.size() + captures.size();
  if (entry.getNumArguments() != expectedArgs) {
    return emitOpError() << "expects " << expectedArgs << " block argument(s) ("
                         << tokens.size() << " token + " << captures.size()
                         << " capture); got " << entry.getNumArguments();
  }
  for (auto [idx, tt] : llvm::enumerate(tokenTypes)) {
    Type argTy = entry.getArgument(idx).getType();
    Type sliceTy = tt.getSliceType();
    if (argTy != sliceTy) {
      return emitOpError() << "block argument #" << idx << " type (" << argTy
                           << ") does not match token slice type (" << sliceTy
                           << ")";
    }
  }
  for (auto [idx, capture] : llvm::enumerate(captures)) {
    if (!isScalarCaptureType(capture.getType()))
      return emitOpError()
             << "capture operand #" << idx
             << " must be an integer, index, or float scalar; got "
             << capture.getType();

    unsigned argIdx = tokens.size() + idx;
    Type argTy = entry.getArgument(argIdx).getType();
    if (argTy != capture.getType()) {
      return emitOpError() << "capture block argument #" << idx << " type ("
                           << argTy << ") does not match capture operand type ("
                           << capture.getType() << ")";
    }
  }

  // Terminator is sde.yield with no values. Memref codelets update through
  // token block arguments directly.
  auto yield = llvm::dyn_cast_or_null<SdeYieldOp>(entry.getTerminator());
  if (!yield)
    return emitOpError() << "expects body to terminate with sde.yield";
  if (!yield.getValues().empty())
    return emitOpError() << "expects memref codelet yield to carry no values";

  // Best-effort check for conflicting modes on statically-overlapping slices
  // of the same source storage value. Non-constant slices are delegated to
  // runtime.
  auto modeKind = [](SdeAccessMode m) {
    // 0 = read, 1 = write-ish (write or readwrite).
    return (m == SdeAccessMode::read) ? 0 : 1;
  };

  auto constantSlice = [](SdeMuTokenOp op, SmallVectorImpl<int64_t> &offs,
                          SmallVectorImpl<int64_t> &szs) -> bool {
    if (op.getOffsets().size() != op.getSizes().size())
      return false;
    for (auto [off, sz] : llvm::zip(op.getOffsets(), op.getSizes())) {
      APInt offV, szV;
      if (!matchPattern(off, m_ConstantInt(&offV)))
        return false;
      if (!matchPattern(sz, m_ConstantInt(&szV)))
        return false;
      offs.push_back(offV.getSExtValue());
      szs.push_back(szV.getSExtValue());
    }
    return true;
  };

  auto slicesOverlap = [](ArrayRef<int64_t> aOff, ArrayRef<int64_t> aSz,
                          ArrayRef<int64_t> bOff, ArrayRef<int64_t> bSz) {
    // Whole-storage tokens (empty offsets/sizes) overlap everything from the
    // same parent.
    if (aOff.empty() || bOff.empty())
      return true;
    if (aOff.size() != bOff.size())
      return true; // degenerate; be conservative.
    for (size_t i = 0; i < aOff.size(); ++i) {
      int64_t aEnd = aOff[i] + aSz[i];
      int64_t bEnd = bOff[i] + bSz[i];
      if (aEnd <= bOff[i] || bEnd <= aOff[i])
        return false;
    }
    return true;
  };

  for (size_t i = 0; i < tokens.size(); ++i) {
    auto ai = tokens[i].getDefiningOp<SdeMuTokenOp>();
    if (!ai)
      continue;
    SmallVector<int64_t> aOff, aSz;
    bool aConst = constantSlice(ai, aOff, aSz);
    for (size_t j = i + 1; j < tokens.size(); ++j) {
      auto bi = tokens[j].getDefiningOp<SdeMuTokenOp>();
      if (!bi)
        continue;
      if (ai.getSource() != bi.getSource())
        continue;
      if (modeKind(ai.getMode()) == modeKind(bi.getMode()) &&
          ai.getMode() == bi.getMode())
        continue; // identical mode: always fine.
      // read+read already filtered above; read+write or write+write require
      // disjointness.
      SmallVector<int64_t> bOff, bSz;
      bool bConst = constantSlice(bi, bOff, bSz);
      if (!aConst || !bConst)
        continue; // delegate to runtime.
      if (slicesOverlap(aOff, aSz, bOff, bSz)) {
        return emitOpError()
               << "tokens #" << i << " and #" << j
               << " have conflicting access modes on statically-overlapping "
                  "slices of the same source storage";
      }
    }
  }

  // TODO: single-use token enforcement (dedicated pattern, not a verifier).
  return success();
}

LogicalResult SdeMuReductionDeclOp::verify() {
  if (auto identity = getIdentity()) {
    auto typedIdentity = dyn_cast<TypedAttr>(*identity);
    if (!typedIdentity)
      return emitOpError() << "expects identity to be a typed attribute";
    if (typedIdentity.getType() != getType())
      return emitOpError() << "expects identity type to match reduction type";
  }

  if (getReductionKind() != SdeReductionKind::custom)
    return success();

  if (getCombiner().empty())
    return emitOpError() << "expects a combiner region for custom reductions";

  Block &entry = getCombiner().front();
  if (entry.getNumArguments() != 2)
    return emitOpError()
           << "expects custom combiner region with two block arguments";
  if (entry.getArgumentTypes()[0] != getType() ||
      entry.getArgumentTypes()[1] != getType())
    return emitOpError()
           << "expects custom combiner arguments to match reduction type";

  auto yield = dyn_cast<SdeYieldOp>(entry.getTerminator());
  if (!yield || yield.getValues().size() != 1 ||
      yield.getValues().front().getType() != getType())
    return emitOpError() << "expects custom combiner to terminate with "
                            "sde.yield of the reduction type";

  return success();
}
