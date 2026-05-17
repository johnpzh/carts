///==========================================================================///
/// File: Compile.cpp
/// Main entry point for the CARTS compilation pipeline tool.
///==========================================================================///

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "polygeist/Dialect.h"
#include "polygeist/Ops.h"
#include "polygeist/Passes/Passes.h"

#include "carts/dialect/arts/IR/ArtsDialect.h"
#include "carts/dialect/arts/Analysis/AnalysisManager.h"
#include "carts/dialect/arts-rt/IR/RtDialect.h"
#include "carts/dialect/sde/IR/SdeDialect.h"
#include "carts/passes/Passes.h"
#include "carts/utils/Debug.h"
#include "carts/utils/OperationAttributes.h"
#include "carts/utils/PassInstrumentation.h"
#include "carts/dialect/codir/Conversion/PassRegistration.h"
#include "carts/dialect/codir/IR/CodirDialect.h"
#include "carts/dialect/codir/Transforms/PassRegistration.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include <array>
#include <cassert>
#include <functional>
#include <optional>
#include <string>
#include <vector>

using namespace llvm;
using namespace mlir;
using namespace mlir::carts;
using mlir::carts::debugStream;

ARTS_DEBUG_SETUP(compile)

namespace {
constexpr const char *kDefaultDiagnoseOutput = ".carts-diagnose.json";
constexpr uint64_t kDefaultArtsIdStride = 1000;
} // namespace

///===----------------------------------------------------------------------===///
/// Interface Attachments
///===----------------------------------------------------------------------===///
/// Use the original models for attaching type interfaces.
class MemRefInsider
    : public MemRefElementTypeInterface::FallbackModel<MemRefInsider> {};

template <typename T>
struct PtrElementModel
    : public LLVM::PointerElementTypeInterface::ExternalModel<
          PtrElementModel<T>, T> {};

///===----------------------------------------------------------------------===///
/// Command Line Options
///===----------------------------------------------------------------------===///
static cl::opt<std::string>
    InputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string> OutputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

static cl::opt<bool> Opt("O3", cl::desc("Apply Optimizations"),
                         cl::init(false));

static cl::opt<bool> EmitLLVM("emit-llvm", cl::desc("Emit LLVM IR output"),
                              cl::init(false));

static cl::opt<bool> Debug("g", cl::desc("Enable debug mode"), cl::init(false));

static cl::opt<std::string> ArtsConfig("arts-config",
                                       cl::desc("ARTS configuration file path"),
                                       cl::value_desc("config_file"),
                                       cl::init(""));

static cl::opt<uint64_t>
    ArtsIdStride("arts-id-stride",
                 cl::desc("Stride multiplier for arts ids (EDTs/DBs)"),
                 cl::init(kDefaultArtsIdStride));

static cl::opt<bool>
    Diagnose("diagnose",
             cl::desc("Export diagnostic information about compilation"),
             cl::init(false));

static cl::opt<std::string> DiagnoseOutput(
    "diagnose-output", cl::desc("Output file for diagnostic JSON export"),
    cl::value_desc("filename"), cl::init(kDefaultDiagnoseOutput));

static cl::opt<bool>
    PassTiming("pass-timing",
               cl::desc("Print per-pass timing report after compilation"),
               cl::init(false));

static cl::opt<std::string>
    PassTimingOutput("pass-timing-output",
                     cl::desc("Export per-pass timing data as JSON to a file"),
                     cl::value_desc("filename"), cl::init(""));

static cl::opt<std::string> ArtsDebug(
    "arts-debug",
    cl::desc("Enable ARTS_INFO/ARTS_DEBUG channels (comma-separated)"),
    cl::value_desc("debug_types"), cl::init(""));

static cl::opt<bool> RuntimeStaticWorkers(
    "runtime-static-workers",
    cl::desc("Fold runtime_query<total_workers> to the configured cluster-wide "
             "worker count when the module embeds a valid ARTS config"),
    cl::init(false));

/// Distributed DB allocation enablement.
static cl::opt<bool> DistributedDb(
    "distributed-db",
    cl::desc("Attempt distributed DB allocation "
             "(ownership marking + parallel initPerWorker creation)"),
    cl::init(false));

///===----------------------------------------------------------------------===///
/// Pipeline Stop Options
///===----------------------------------------------------------------------===///
enum class StageId {
  SdeInputNormalization,
  InitialCleanup,
  SdePlanning,
  SdeToCodir,
  CodirToArts,
  EdtTransforms,
  CreateDbs,
  DbOpt,
  PostDbRefinement,
  LateConcurrencyCleanup,
  Epochs,
  PreLowering,
  ArtsRtToLLVM,
  PostO3Opt,
  LLVMIREmission
};

enum class StageKind { Core, Epilogue };

struct StageExecutionContext {
  ModuleOp module;
  MLIRContext &context;
  arts::AnalysisManager *analysisManager = nullptr;
  const arts::RuntimeConfig *machine = nullptr;
  bool stopAfterStage = false;
  bool runAdditionalOpt = false;
  bool emitLLVM = false;
};

using StageBuilderFn = void (*)(PassManager &, const StageExecutionContext &);
using StageEnabledFn = bool (*)(const StageExecutionContext &);

struct StageDescriptor {
  StageId id;
  llvm::StringLiteral token;
  StageKind kind;
  bool allowPipelineStop;
  bool allowStartFrom;
  bool captureDiagnosticsBefore;
  llvm::StringLiteral errorMessage;
  llvm::ArrayRef<llvm::StringLiteral> passes;
  StageBuilderFn build;
  StageEnabledFn enabled;
  /// Stage tokens that must run before this stage. Used for --start-from
  /// validation and pipeline ordering verification.
  llvm::ArrayRef<llvm::StringLiteral> dependsOn;
};

struct DialectGroupDescriptor {
  llvm::StringLiteral name;
  llvm::StringLiteral status;
  llvm::StringLiteral summary;
  llvm::ArrayRef<llvm::StringLiteral> layers;
  llvm::ArrayRef<llvm::StringLiteral> currentStages;
  llvm::ArrayRef<llvm::StringLiteral> targetStages;
};

static constexpr llvm::StringLiteral kCompletePipelineToken = "complete";
static constexpr llvm::StringLiteral kPostO3OptToken = "post-o3-opt";
static constexpr llvm::StringLiteral kLLVMIREmissionToken = "llvm-ir-emission";
static constexpr llvm::StringLiteral kArtsRtToLLVMToken = "arts-rt-to-llvm";

static cl::opt<std::string> Pipeline(
    "pipeline",
    cl::desc("Stop pipeline at specified stage token "
             "(use --print-pipeline-manifest-json to inspect valid tokens)"),
    cl::value_desc("stage"), cl::init("complete"));

static cl::opt<std::string> StartFrom(
    "start-from",
    cl::desc("Resume pipeline from specified stage token "
             "(use --print-pipeline-manifest-json to inspect valid tokens)"),
    cl::value_desc("stage"), cl::init("sde-input-normalization"));

static cl::opt<bool> PrintPipelineManifestJSON(
    "print-pipeline-manifest-json",
    cl::desc("Print pipeline step/pass manifest as JSON and exit"),
    cl::init(false));

static cl::opt<std::string> CustomPassPipeline(
    "pass-pipeline",
    cl::desc("Run a textual MLIR pass pipeline instead of the staged CARTS "
             "pipeline"),
    cl::value_desc("pipeline"), cl::init(""));

static const std::array<llvm::StringLiteral, 10>
    kSdeInputNormalizationPasses = {"LowerAffine(func)",
                                    "CSE",
                                    "SdeInputInliner",
                                    "PolygeistCanonicalize",
                                    "ScalarForwarding",
                                    "PolygeistCanonicalize",
                                    "SdeMemrefNormalization",
                                    "SdeHandleDeps",
                                    "SdeDeadStateCleanup",
                                    "CSE"};
static const std::array<llvm::StringLiteral, 3> kInitialCleanupPasses = {
    "LowerAffine(func)", "CSE(func)", "PolygeistCanonicalizeFor(func)"};
static const std::array<llvm::StringLiteral, 14> kSdePlanningPasses = {
    "ConvertOpenMPToSde",
    "PatternAnalysis",
    "LoopInterchange",
    "Tiling",
    "ElementwiseFusion",
    "Vectorization",
    "ScheduleRefinement",
    "ChunkOpt",
    "ReductionStrategy",
    "DistributionPlanning",
    "IterationSpaceDecomposition",
    "BarrierElimination",
    "VerifySdeCpsPlan",
    "MemoryUnitMaterialization"};
static const std::array<llvm::StringLiteral, 3> kSdeToCodirPasses = {
    "ConvertSdeToCodir", "CodirCodeletOpt", "VerifyCodir"};
static const std::array<llvm::StringLiteral, 6> kCodirToArtsPasses = {
    "ConvertCodirToArts",
    "VerifySdeLowered",
    "VerifyArtsObjectsOnly",
    "ArtsDeadCodeElimination",
    "CSE(arts.edt)",
    "VerifyEdtCreated"};
static const std::array<llvm::StringLiteral, 5> kEdtTransformsPasses = {
    "EdtStructuralOpt(runAnalysis=false)",
    "ArtsDeadCodeElimination",
    "SymbolDCE",
    "CSE(arts.edt)",
    "EdtPtrRematerialization"};
static const std::array<llvm::StringLiteral, 6> kCreateDbsPasses = {
    "CreateDbs", "PolygeistCanonicalize", "CSE(arts.edt)",
    "SymbolDCE", "Mem2Reg", "PolygeistCanonicalize"};
static const std::array<llvm::StringLiteral, 4> kDbOptPasses = {
    "DbModeTightening", "PolygeistCanonicalize", "CSE(arts.edt)", "Mem2Reg"};
static const std::array<llvm::StringLiteral, 8> kPostDbRefinementPasses = {
    "DbModeTightening",
    "DbDistributedOwnership (conditional)",
    "EdtTransforms",
    "DbTransforms",
    "ContractValidation",
    "DbScratchElimination",
    "PolygeistCanonicalize",
    "CSE(arts.edt)"};
static const std::array<llvm::StringLiteral, 7> kLateConcurrencyCleanupPasses =
    {"BlockLoopStripMining(func)",
     "Hoisting",
     "PolygeistCanonicalize",
     "CSE(arts.edt)",
     "EdtAllocaSinking",
     "ArtsDeadCodeElimination",
     "Mem2Reg"};
static const std::array<llvm::StringLiteral, 5> kEpochsPasses = {
    "PolygeistCanonicalize", "CreateEpochs",
    "VerifyEpochCreated", "EpochOpt[scheduling] (conditional)",
    "PolygeistCanonicalize"};
static const std::array<llvm::StringLiteral, 22> kPreLoweringPasses = {
    "EdtAllocaSinking",
    "PolygeistCanonicalize",
    "CSE(arts.edt)",
    "DbLowering",
    "PolygeistCanonicalize",
    "CSE(arts.edt)",
    "EdtLowering",
    "PolygeistCanonicalize",
    "CSE",
    "VerifyEdtLowered",
    "LICM",
    "DataPtrHoisting",
    "PolygeistCanonicalize",
    "CSE",
    "ScalarReplacement",
    "PolygeistCanonicalize",
    "CSE",
    "EpochLowering",
    "PolygeistCanonicalize",
    "CSE",
    "VerifyEpochLowered",
    "VerifyPreLowered"};
static const std::array<llvm::StringLiteral, 14> kArtsRtToLLVMPasses = {
    "LowerAffine(func)",
    "ConvertArtsRtToLLVM",
    "LoweringContractCleanup",
    "GuidRangeCallOpt",
    "RuntimeCallOpt",
    "DataPtrHoisting",
    "PolygeistCanonicalize",
    "CSE",
    "Mem2Reg",
    "PolygeistCanonicalize",
    "ControlFlowSink",
    "PolygeistCanonicalize",
    "VerifyDbLowered",
    "VerifyLowered"};

static const std::array<llvm::StringLiteral, 6> kPostO3OptPasses = {
    "PolygeistCanonicalize",
    "ControlFlowSink",
    "PolygeistCanonicalize",
    "LICM",
    "CSE",
    "PolygeistCanonicalize"};
static const std::array<llvm::StringLiteral, 15> kLLVMIREmissionPasses = {
    "CSE",
    "PolygeistCanonicalize",
    "MaterializeArtsFunctionPointers",
    "ConvertOpenMPToLLVM",
    "ArithExpandOps",
    "ConvertSCFToCF",
    "ResidualHostOpenMPMemrefCleanup",
    "ConvertPolygeistToLLVM",
    "ConvertIndexToLLVM",
    "ConvertControlFlowToLLVM",
    "ReconcileUnrealizedCasts",
    "AliasScopeGen",
    "LoopVectorizationHints",
    "PolygeistCanonicalize",
    "CSE"};

static constexpr llvm::StringLiteral kFrontendLayers[] = {
    "polygeist", "memref", "scf"};
static constexpr llvm::StringLiteral kArtsLayers[] = {"arts"};
static constexpr llvm::StringLiteral kArtsRtLayers[] = {"arts-rt", "llvm"};
static constexpr llvm::StringLiteral kSdeLayers[] = {"sde"};
static constexpr llvm::StringLiteral kCodirLayers[] = {"codir"};
static constexpr llvm::StringLiteral kSdeToCodirLayers[] = {"sde", "codir"};
static constexpr llvm::StringLiteral kCodirToArtsLayers[] = {"codir", "arts"};

static constexpr llvm::StringLiteral kCurrentFrontendStages[] = {
    "sde-input-normalization", "initial-cleanup"};
static constexpr llvm::StringLiteral kCurrentSdePlanningStages[] = {
    "sde-planning"};
static constexpr llvm::StringLiteral kCurrentSdeToCodirStages[] = {
    "sde-to-codir"};
static constexpr llvm::StringLiteral kCurrentCodirToArtsStages[] = {
    "codir-to-arts"};
static constexpr llvm::StringLiteral kCurrentArtsStages[] = {
    "edt-transforms", "create-dbs", "db-opt", "post-db-refinement",
    "late-concurrency-cleanup", "epochs"};
static constexpr llvm::StringLiteral kCurrentArtsRtStages[] = {
    "pre-lowering", "arts-rt-to-llvm"};

static constexpr llvm::StringLiteral kTargetFrontendStages[] = {
    "frontend-normalization"};
static constexpr llvm::StringLiteral kTargetSdeStages[] = {
    "openmp-to-sde", "sde-planning"};
static constexpr llvm::StringLiteral kTargetSdeToCodirStages[] = {
    "sde-to-codir"};
static constexpr llvm::StringLiteral kTargetCodirStages[] = {
    "verify-codir", "codir-codelet-opt"};
static constexpr llvm::StringLiteral kTargetCodirToArtsStages[] = {
    "codir-to-arts"};
static constexpr llvm::StringLiteral kTargetArtsStages[] = {
    "arts-object-refinement", "arts-epochs"};
static constexpr llvm::StringLiteral kTargetArtsRtStages[] = {
    "pre-lowering", "arts-rt-to-llvm"};

static const std::array<DialectGroupDescriptor, 6> kCurrentDialectGroups = {{
    {"frontend-normalization", "current",
     "Frontend and memref normalization before SDE owns source semantics.",
     kFrontendLayers, kCurrentFrontendStages, kTargetFrontendStages},
    {"sde", "current",
     "SDE proves source semantics and authors MU/CU/SU planning facts.",
     kSdeLayers, kCurrentSdePlanningStages, kTargetSdeStages},
    {"sde-to-codir", "current",
     "Materialize SDE codelet plans into isolated CODIR codelets.",
     kSdeToCodirLayers, kCurrentSdeToCodirStages, kTargetSdeToCodirStages},
    {"codir-to-arts", "current",
     "Lower CODIR codelet deps and params to ARTS DB/EDT objects; no SDE operation may survive this boundary.",
     kCodirToArtsLayers, kCurrentCodirToArtsStages, kTargetCodirToArtsStages},
    {"arts-object-refinement", "current",
     "Abstract ARTS DB, EDT, epoch, dependency, and cleanup stages.",
     kArtsLayers, kCurrentArtsStages, kTargetArtsStages},
    {"arts-rt-lowering", "current",
     "Runtime ABI and LLVM-facing lowering after ARTS object shape is chosen.",
     kArtsRtLayers, kCurrentArtsRtStages, kTargetArtsRtStages},
}};

static const std::array<DialectGroupDescriptor, 7> kTargetDialectGroups = {{
    {"frontend-normalization", "target",
     "Normalize frontend IR before entering the CARTS dialect stack.",
     kFrontendLayers, kCurrentFrontendStages, kTargetFrontendStages},
    {"sde", "target",
     "SDE proves source semantics and authors MU/CU/SU planning facts.",
     kSdeLayers, kCurrentSdePlanningStages, kTargetSdeStages},
    {"sde-to-codir", "current",
     "Materialize SDE plans into isolated CODIR codelets and token-local views.",
     kSdeToCodirLayers, kCurrentSdeToCodirStages, kTargetSdeToCodirStages},
    {"codir", "current",
     "Verify explicit deps, params, yielded values, and no implicit captures.",
     kCodirLayers, kCurrentSdeToCodirStages, kTargetCodirStages},
    {"codir-to-arts", "current",
     "Lower explicit CODIR deps and codelets to ARTS DB/EDT objects.",
     kCodirToArtsLayers, kCurrentCodirToArtsStages, kTargetCodirToArtsStages},
    {"arts", "target",
     "Refine abstract ARTS DB, EDT, epoch, dependency, and placement objects.",
     kArtsLayers, kCurrentArtsStages, kTargetArtsStages},
    {"arts-rt", "target",
     "Lower the chosen ARTS object graph to runtime ABI and LLVM.",
     kArtsRtLayers, kCurrentArtsRtStages, kTargetArtsRtStages},
}};

static ArrayRef<StageDescriptor> getStageRegistry();

static const StageDescriptor *findStageById(StageId id) {
  for (const auto &stage : getStageRegistry()) {
    if (stage.id == id)
      return &stage;
  }
  return nullptr;
}

static const StageDescriptor *findStageByToken(StringRef token) {
  for (const auto &stage : getStageRegistry()) {
    if (stage.token == token)
      return &stage;
  }
  return nullptr;
}

static int stageIndex(StageId id, StageKind kind) {
  int index = 0;
  for (const auto &stage : getStageRegistry()) {
    if (stage.kind != kind)
      continue;
    if (stage.id == id)
      return index;
    ++index;
  }
  return -1;
}

static llvm::StringRef stageName(StageId id) {
  if (const auto *stage = findStageById(id))
    return stage->token;
  return "unknown";
}

static bool shouldIncludeStageInJSON(const StageDescriptor &stage,
                                     llvm::StringRef key) {
  if (key == "pipeline")
    return stage.allowPipelineStop;
  if (key == "start_from")
    return stage.allowStartFrom;
  return stage.kind == StageKind::Core;
}

template <typename Predicate>
static void printPipelineTokenArray(llvm::raw_ostream &os, llvm::StringRef key,
                                    Predicate include) {
  os << "  \"" << key << "\": [";
  bool first = true;
  for (const auto &stage : getStageRegistry()) {
    if (!include(stage))
      continue;
    if (!first)
      os << ", ";
    first = false;
    os << "\"" << stage.token << "\"";
  }
  if (key == "pipeline") {
    if (!first)
      os << ", ";
    os << "\"" << kCompletePipelineToken << "\"";
  }
  os << "]";
}

static void printStringArray(llvm::raw_ostream &os,
                             llvm::ArrayRef<llvm::StringLiteral> values) {
  os << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0)
      os << ", ";
    os << "\"" << values[i] << "\"";
  }
  os << "]";
}

static void printDialectGroupArray(
    llvm::raw_ostream &os,
    llvm::ArrayRef<DialectGroupDescriptor> groups) {
  os << "[\n";
  for (size_t i = 0; i < groups.size(); ++i) {
    const DialectGroupDescriptor &group = groups[i];
    if (i != 0)
      os << ",\n";
    os << "      {\"name\": \"" << group.name << "\", \"status\": \""
       << group.status << "\", \"summary\": \"" << group.summary
       << "\", \"layers\": ";
    printStringArray(os, group.layers);
    os << ", \"currentStages\": ";
    printStringArray(os, group.currentStages);
    os << ", \"targetStages\": ";
    printStringArray(os, group.targetStages);
    os << "}";
  }
  os << "\n    ]";
}

static void printPipelineManifestAsJSON(llvm::raw_ostream &os) {
  os << "{\n";
  printPipelineTokenArray(os, "pipeline", [](const StageDescriptor &stage) {
    return shouldIncludeStageInJSON(stage, "pipeline");
  });
  os << ",\n";
  printPipelineTokenArray(os, "start_from", [](const StageDescriptor &stage) {
    return shouldIncludeStageInJSON(stage, "start_from");
  });
  os << ",\n";
  printPipelineTokenArray(
      os, "pipeline_sequence", [](const StageDescriptor &stage) {
        return shouldIncludeStageInJSON(stage, "pipeline_sequence");
      });
  os << ",\n";
  os << "  \"pipeline_steps\": [\n";
  bool firstStep = true;
  for (const auto &stage : getStageRegistry()) {
    if (stage.kind != StageKind::Core)
      continue;
    if (!firstStep)
      os << ",\n";
    firstStep = false;
    os << "    {\"name\": \"" << stage.token << "\", \"passes\": ";
    printStringArray(os, stage.passes);
    os << ", \"dependsOn\": ";
    printStringArray(os, stage.dependsOn);
    os << "}";
  }
  os << "\n  ],\n";
  os << "  \"epilogue_steps\": [\n";
  bool firstEpilogue = true;
  for (const auto &stage : getStageRegistry()) {
    if (stage.kind != StageKind::Epilogue)
      continue;
    if (!firstEpilogue)
      os << ",\n";
    firstEpilogue = false;
    os << "    {\"name\": \"" << stage.token << "\", \"passes\": ";
    printStringArray(os, stage.passes);
    os << ", \"dependsOn\": ";
    printStringArray(os, stage.dependsOn);
    os << "}";
  }
  os << "\n  ],\n";
  os << "  \"dialect_groups\": {\n";
  os << "    \"current\": ";
  printDialectGroupArray(os, kCurrentDialectGroups);
  os << ",\n";
  os << "    \"target\": ";
  printDialectGroupArray(os, kTargetDialectGroups);
  os << "\n  }\n";
  os << "}\n";
}

static void configureArtsDebugChannels(llvm::StringRef channels) {
  if (channels.empty())
    return;

  llvm::SmallVector<llvm::StringRef, 8> splitChannels;
  channels.split(splitChannels, ',', -1, false);

  static llvm::SmallVector<std::string, 8> ownedChannels;
  ownedChannels.clear();
  ownedChannels.reserve(splitChannels.size());
  for (llvm::StringRef channel : splitChannels) {
    llvm::StringRef trimmed = channel.trim();
    if (!trimmed.empty())
      ownedChannels.push_back(trimmed.str());
  }
  if (ownedChannels.empty())
    return;

  llvm::DebugFlag = true;
  llvm::SmallVector<const char *, 8> debugTypes;
  debugTypes.reserve(ownedChannels.size());
  for (const std::string &owned : ownedChannels)
    debugTypes.push_back(owned.c_str());
  llvm::setCurrentDebugTypes(debugTypes.data(), debugTypes.size());
}

static bool shouldExportDetailedDiagnose(std::optional<StageId> stopAt) {
  if (!stopAt)
    return true;
  int stopIndex = stageIndex(*stopAt, StageKind::Core);
  int preLoweringIndex = stageIndex(StageId::PreLowering, StageKind::Core);
  return stopIndex >= 0 && preLoweringIndex >= 0 &&
         stopIndex >= preLoweringIndex;
}

static LogicalResult
configurePassManager(PassManager &pm,
                     arts::PassTimingData *timingData = nullptr) {
  pm.enableVerifier(true);
  if (failed(applyPassManagerCLOptions(pm)))
    return failure();
  applyDefaultTimingPassManagerCLOptions(pm);
  if (timingData)
    pm.addInstrumentation(
        std::make_unique<arts::CartsPassInstrumentation>(timingData));
  return success();
}

///===----------------------------------------------------------------------===///
/// Helper Functions for Initialization and Pass Setup
///===----------------------------------------------------------------------===///
/// Register standard MLIR dialects, passes, and translations.
void registerDialects(DialectRegistry &registry) {
  registry.insert<polygeist::PolygeistDialect, arts::ArtsDialect,
                  arts_rt::ArtsRtDialect, sde::CartsSdeDialect,
                  codir::CartsCodirDialect>();
  registerAllPasses();
  /// ARTS pass registration is intentionally selective: several ARTS passes
  /// require the staged pipeline's AnalysisManager and cannot be constructed
  /// through textual pass registration with default arguments.
  registerDeadCodeElimination();
  registerVerifyArtsObjectsOnly();
  registerArtsRtPasses();
  sde::registerCartsSdePasses();
  codir::registerCartsCodirConversionPasses();
  codir::registerCartsCodirPasses();
  registerAllTranslations();
  registerpolygeistPasses();
  func::registerInlinerExtension(registry);
  registerAllDialects(registry);
  registerAllExtensions(registry);
  registerAllFromLLVMIRTranslations(registry);
  registerAllToLLVMIRTranslations(registry);
}

/// Initialize the MLIR context by loading necessary dialects and attaching
/// type interfaces.
void initializeContext(MLIRContext &context) {
  context.disableMultithreading(true);
  context.getOrLoadDialect<affine::AffineDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<DLTIDialect>();
  context.getOrLoadDialect<scf::SCFDialect>();
  context.getOrLoadDialect<async::AsyncDialect>();
  context.getOrLoadDialect<LLVM::LLVMDialect>();
  context.getOrLoadDialect<NVVM::NVVMDialect>();
  context.getOrLoadDialect<ROCDL::ROCDLDialect>();
  context.getOrLoadDialect<gpu::GPUDialect>();
  context.getOrLoadDialect<mlir::omp::OpenMPDialect>();
  context.getOrLoadDialect<math::MathDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  context.getOrLoadDialect<linalg::LinalgDialect>();
  context.getOrLoadDialect<tensor::TensorDialect>();
  context.getOrLoadDialect<bufferization::BufferizationDialect>();
  context.getOrLoadDialect<polygeist::PolygeistDialect>();
  context.getOrLoadDialect<arts::ArtsDialect>();
  context.getOrLoadDialect<arts_rt::ArtsRtDialect>();
  context.getOrLoadDialect<sde::CartsSdeDialect>();
  context.getOrLoadDialect<codir::CartsCodirDialect>();
  context.getOrLoadDialect<cf::ControlFlowDialect>();

  /// Register all necessary interfaces for LLVM conversion
  LLVM::LLVMFunctionType::attachInterface<MemRefInsider>(context);
  LLVM::LLVMArrayType::attachInterface<MemRefInsider>(context);
  LLVM::LLVMPointerType::attachInterface<MemRefInsider>(context);
  LLVM::LLVMStructType::attachInterface<MemRefInsider>(context);
  MemRefType::attachInterface<PtrElementModel<MemRefType>>(context);
  IndexType::attachInterface<PtrElementModel<IndexType>>(context);
  LLVM::LLVMStructType::attachInterface<PtrElementModel<LLVM::LLVMStructType>>(
      context);
  LLVM::LLVMPointerType::attachInterface<
      PtrElementModel<LLVM::LLVMPointerType>>(context);
  LLVM::LLVMArrayType::attachInterface<PtrElementModel<LLVM::LLVMArrayType>>(
      context);
}

static void ensureRuntimeConfigDataVisibleForValidation(ModuleOp module) {
  std::optional<StringRef> configData = arts::getRuntimeConfigData(module);
  if (!configData || configData->empty())
    return;
  constexpr llvm::StringLiteral kConfigGlobalName =
      "__carts_embedded_arts_cfg";
  if (module.lookupSymbol<LLVM::GlobalOp>(kConfigGlobalName))
    return;

  OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  auto i8 = mlir::IntegerType::get(module.getContext(), 8);
  auto type = LLVM::LLVMArrayType::get(i8, configData->size() + 1);
  LLVM::GlobalOp::create(
      builder, module.getLoc(), type, /*isConstant=*/true,
      LLVM::Linkage::Internal, kConfigGlobalName,
      builder.getStringAttr(configData->str() + '\0'));
}

static bool hasResidualOpenMP(ModuleOp module) {
  bool found = false;
  module.walk([&](Operation *op) {
    if (op->getDialect() && op->getDialect()->getNamespace() == "omp") {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static void markHostOpenMPBenchmarkMode(ModuleOp module) {
  if (!hasResidualOpenMP(module))
    return;

  constexpr llvm::StringLiteral kMarkerFn =
      "carts_benchmarks_mark_host_openmp";
  MLIRContext *ctx = module.getContext();
  if (!module.lookupSymbol<LLVM::LLVMFuncOp>(kMarkerFn)) {
    OpBuilder builder(ctx);
    builder.setInsertionPointToStart(module.getBody());
    auto fnType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                              /*params=*/{}, false);
    auto fn = LLVM::LLVMFuncOp::create(builder, module.getLoc(), kMarkerFn,
                                       fnType);
    fn.setLinkage(LLVM::Linkage::External);
  }

  SmallVector<LLVM::CallOp> benchmarkStarts;
  module.walk([&](LLVM::CallOp call) {
    if (call.getCallee() == "carts_benchmarks_start")
      benchmarkStarts.push_back(call);
  });

  for (LLVM::CallOp call : benchmarkStarts) {
    OpBuilder builder(call);
    LLVM::CallOp::create(builder, call.getLoc(), TypeRange{},
                         SymbolRefAttr::get(ctx, kMarkerFn), ValueRange{});
  }
}

static bool isArtsOutlinedEdtName(StringRef name) {
  return name.starts_with("__arts_edt_");
}

static LogicalResult promoteOutlinedEdtToLLVMFunc(func::FuncOp funcOp,
                                                  OpBuilder &builder) {
  mlir::FunctionType funcType = funcOp.getFunctionType();
  MLIRContext *ctx = funcOp.getContext();

  mlir::Type resultType = LLVM::LLVMVoidType::get(ctx);
  if (funcType.getNumResults() > 1)
    return funcOp.emitError("cannot materialize ARTS EDT function pointer for "
                            "multi-result function");
  if (funcType.getNumResults() == 1) {
    resultType = funcType.getResult(0);
    if (!LLVM::isCompatibleType(resultType))
      return funcOp.emitError("cannot materialize ARTS EDT function pointer "
                              "before LLVM conversion for result type ")
             << resultType;
  }

  SmallVector<mlir::Type, 8> inputTypes;
  inputTypes.reserve(funcType.getNumInputs());
  for (mlir::Type input : funcType.getInputs()) {
    if (!LLVM::isCompatibleType(input))
      return funcOp.emitError("cannot materialize ARTS EDT function pointer "
                              "before LLVM conversion for argument type ")
             << input;
    inputTypes.push_back(input);
  }

  auto llvmType =
      LLVM::LLVMFunctionType::get(resultType, inputTypes, /*isVarArg=*/false);
  builder.setInsertionPoint(funcOp);
  auto llvmFunc = LLVM::LLVMFuncOp::create(
      builder, funcOp.getLoc(), funcOp.getName(), llvmType,
      LLVM::Linkage::External, /*dsoLocal=*/false, LLVM::CConv::C);
  cast<FunctionOpInterface>(llvmFunc.getOperation())
      .setVisibility(funcOp.getVisibility());
  llvmFunc.getBody().takeBody(funcOp.getBody());
  SmallVector<func::ReturnOp, 4> returns;
  llvmFunc.walk([&](func::ReturnOp ret) { returns.push_back(ret); });
  for (func::ReturnOp ret : returns) {
    builder.setInsertionPoint(ret);
    LLVM::ReturnOp::create(builder, ret.getLoc(), ret.getOperands());
    ret.erase();
  }
  funcOp.erase();
  return success();
}

struct MaterializeArtsFunctionPointersPass
    : public PassWrapper<MaterializeArtsFunctionPointersPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MaterializeArtsFunctionPointersPass)

  StringRef getArgument() const final {
    return "carts-materialize-arts-function-pointers";
  }

  StringRef getDescription() const final {
    return "Materialize ARTS EDT function pointer symbols before mixed "
           "host-OpenMP LLVM emission";
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (!hasResidualOpenMP(module))
      return;

    SmallVector<polygeist::GetFuncOp, 8> getFuncOps;
    module.walk([&](polygeist::GetFuncOp op) {
      if (isArtsOutlinedEdtName(op.getName()))
        getFuncOps.push_back(op);
    });
    if (getFuncOps.empty())
      return;

    OpBuilder builder(module.getContext());
    for (polygeist::GetFuncOp getFunc : getFuncOps) {
      StringRef name = getFunc.getName();
      if (!module.lookupSymbol<LLVM::LLVMFuncOp>(name)) {
        auto funcOp = module.lookupSymbol<func::FuncOp>(name);
        if (!funcOp) {
          getFunc.emitError("ARTS EDT function pointer target was not found: ")
              << name;
          signalPassFailure();
          return;
        }
        if (failed(promoteOutlinedEdtToLLVMFunc(funcOp, builder))) {
          signalPassFailure();
          return;
        }
      }

      builder.setInsertionPoint(getFunc);
      auto address = LLVM::AddressOfOp::create(builder, getFunc.getLoc(),
                                               getFunc.getType(), name);
      getFunc.getResult().replaceAllUsesWith(address.getResult());
      getFunc.erase();
    }
  }
};

static bool isLLVMMemRefDescriptorLike(mlir::Type type) {
  auto structType = dyn_cast<LLVM::LLVMStructType>(type);
  if (!structType || structType.getBody().size() < 2)
    return false;
  return isa<LLVM::LLVMPointerType>(structType.getBody()[0]) &&
         isa<LLVM::LLVMPointerType>(structType.getBody()[1]);
}

static void foldResidualHostOpenMPMemrefPointerCasts(ModuleOp module) {
  SmallVector<UnrealizedConversionCastOp> casts;
  module.walk([&](UnrealizedConversionCastOp cast) {
    if (cast.getNumOperands() != 1 || cast.getNumResults() != 1)
      return;
    if (!isa<LLVM::LLVMPointerType>(cast.getResult(0).getType()))
      return;
    if (!isa<MemRefType>(cast.getOperand(0).getType()))
      return;
    casts.push_back(cast);
  });

  for (UnrealizedConversionCastOp cast : casts) {
    if (!cast || cast->use_empty())
      continue;
    auto sourceCast =
        cast.getOperand(0).getDefiningOp<UnrealizedConversionCastOp>();
    if (!sourceCast || sourceCast.getNumOperands() != 1 ||
        sourceCast.getNumResults() != 1)
      continue;
    mlir::Value descriptor = sourceCast.getOperand(0);
    if (!isLLVMMemRefDescriptorLike(descriptor.getType()))
      continue;

    OpBuilder builder(cast);
    mlir::Value ptr = LLVM::ExtractValueOp::create(
        builder, cast.getLoc(), descriptor, /*position=*/1);
    if (ptr.getType() != cast.getResult(0).getType())
      ptr = LLVM::AddrSpaceCastOp::create(
          builder, cast.getLoc(), cast.getResult(0).getType(), ptr);
    cast.getResult(0).replaceAllUsesWith(ptr);
    cast.erase();
    if (sourceCast->use_empty())
      sourceCast.erase();
  }
}

static mlir::Value createLLVMIndexConstant(OpBuilder &builder, Location loc,
                                           mlir::Type type, int64_t value) {
  return LLVM::ConstantOp::create(builder, loc, type,
                                  builder.getIntegerAttr(type, value));
}

static mlir::Value buildRankOneDescriptorFromBarePtr(OpBuilder &builder,
                                                     Location loc,
                                                     mlir::Value ptr,
                                                     mlir::Type descriptorType) {
  auto structType = dyn_cast<LLVM::LLVMStructType>(descriptorType);
  if (!structType || structType.getBody().size() < 5)
    return {};

  mlir::Type indexType = structType.getBody()[2];
  mlir::Value zero = createLLVMIndexConstant(builder, loc, indexType, 0);
  mlir::Value one = createLLVMIndexConstant(builder, loc, indexType, 1);

  mlir::Value descriptor = LLVM::PoisonOp::create(builder, loc, descriptorType);
  descriptor = LLVM::InsertValueOp::create(builder, loc, descriptor, ptr,
                                           ArrayRef<int64_t>{0});
  descriptor = LLVM::InsertValueOp::create(builder, loc, descriptor, ptr,
                                           ArrayRef<int64_t>{1});
  descriptor = LLVM::InsertValueOp::create(builder, loc, descriptor, zero,
                                           ArrayRef<int64_t>{2});
  descriptor = LLVM::InsertValueOp::create(
      builder, loc, descriptor, zero, ArrayRef<int64_t>{3, 0});
  descriptor = LLVM::InsertValueOp::create(
      builder, loc, descriptor, one, ArrayRef<int64_t>{4, 0});
  return descriptor;
}

static void cleanupResidualHostOpenMPMemrefs(ModuleOp module) {
  SmallVector<UnrealizedConversionCastOp> castsToDescriptor;
  module.walk([&](UnrealizedConversionCastOp cast) {
    if (cast.getNumOperands() != 1 || cast.getNumResults() != 1)
      return;
    if (!isLLVMMemRefDescriptorLike(cast.getResult(0).getType()))
      return;
    auto ptrToMemref =
        cast.getOperand(0).getDefiningOp<polygeist::Pointer2MemrefOp>();
    if (!ptrToMemref)
      return;
    auto memrefType = dyn_cast<MemRefType>(ptrToMemref.getType());
    if (!memrefType || memrefType.getRank() != 1 || memrefType.hasStaticShape())
      return;
    castsToDescriptor.push_back(cast);
  });

  for (UnrealizedConversionCastOp cast : castsToDescriptor) {
    if (!cast)
      continue;
    auto ptrToMemref =
        cast.getOperand(0).getDefiningOp<polygeist::Pointer2MemrefOp>();
    if (!ptrToMemref)
      continue;
    OpBuilder builder(cast);
    mlir::Value descriptor = buildRankOneDescriptorFromBarePtr(
        builder, cast.getLoc(), ptrToMemref.getSource(),
        cast.getResult(0).getType());
    if (!descriptor)
      continue;
    cast.getResult(0).replaceAllUsesWith(descriptor);
    cast.erase();
    if (ptrToMemref->use_empty())
      ptrToMemref.erase();
  }

  SmallVector<polygeist::Memref2PointerOp> memrefToPointerOps;
  module.walk([&](polygeist::Memref2PointerOp op) {
    auto sourceCast = op.getSource().getDefiningOp<UnrealizedConversionCastOp>();
    if (!sourceCast || sourceCast.getNumOperands() != 1 ||
        !isLLVMMemRefDescriptorLike(sourceCast.getOperand(0).getType()))
      return;
    memrefToPointerOps.push_back(op);
  });

  for (polygeist::Memref2PointerOp op : memrefToPointerOps) {
    if (!op)
      continue;
    auto sourceCast = op.getSource().getDefiningOp<UnrealizedConversionCastOp>();
    if (!sourceCast)
      continue;
    OpBuilder builder(op);
    mlir::Value ptr = LLVM::ExtractValueOp::create(
        builder, op.getLoc(), sourceCast.getOperand(0), /*position=*/1);
    if (ptr.getType() != op.getType())
      ptr = LLVM::AddrSpaceCastOp::create(builder, op.getLoc(), op.getType(),
                                          ptr);
    op.getResult().replaceAllUsesWith(ptr);
    op.erase();
    if (sourceCast->use_empty())
      sourceCast.erase();
  }
}

struct ResidualHostOpenMPMemrefCleanupPass
    : public PassWrapper<ResidualHostOpenMPMemrefCleanupPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ResidualHostOpenMPMemrefCleanupPass)

  StringRef getArgument() const final {
    return "carts-residual-host-openmp-memref-cleanup";
  }

  StringRef getDescription() const final {
    return "Clean residual host-OpenMP memref/pointer bridges before LLVM "
           "emission";
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (!hasResidualOpenMP(module))
      return;
    cleanupResidualHostOpenMPMemrefs(module);
  }
};

static bool hasArtsRuntimeCallsOutsideHostWrappers(llvm::Module &module) {
  for (llvm::Function &function : module) {
    if (function.isDeclaration())
      continue;
    if (function.getName() == "main" || function.getName() == "main_edt")
      continue;

    for (llvm::BasicBlock &block : function) {
      for (llvm::Instruction &instruction : block) {
        auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction);
        if (!call)
          continue;
        llvm::Function *callee = call->getCalledFunction();
        if (callee && callee->getName().starts_with("arts_"))
          return true;
      }
    }
  }
  return false;
}

static void bypassArtsRuntimeForHostOpenMP(llvm::Module &module) {
  llvm::Function *main = module.getFunction("main");
  llvm::Function *mainBody = module.getFunction("mainBody");
  if (!main || !mainBody || main->arg_size() != 2)
    return;
  if (mainBody->arg_size() != 0 && mainBody->arg_size() != 2)
    return;
  if (hasArtsRuntimeCallsOutsideHostWrappers(module))
    return;

  main->deleteBody();
  llvm::BasicBlock *entry =
      llvm::BasicBlock::Create(module.getContext(), "entry", main);
  llvm::IRBuilder<> builder(entry);
  SmallVector<llvm::Value *, 2> args;
  if (mainBody->arg_size() == 2)
    for (llvm::Argument &arg : main->args())
      args.push_back(&arg);
  llvm::CallInst *result = builder.CreateCall(mainBody, args);
  if (main->getReturnType()->isVoidTy())
    builder.CreateRetVoid();
  else
    builder.CreateRet(result);
}

///===----------------------------------------------------------------------===///
/// Pipeline Builders
///===----------------------------------------------------------------------===///
/// Each build*Pipeline function populates a PassManager with the passes for
/// one logical compilation step. There is a 1:1 mapping between
/// PipelineStep enum values and these builders.

static void addCanonicalizeAndCSE(PassManager &pm) {
  pm.addPass(polygeist::createPolygeistCanonicalizePass());
  pm.addPass(createCSEPass());
}

static void addEdtLocalCSE(PassManager &pm) {
  /// Generic module CSE can replace an EDT-local scalar with an equivalent
  /// outer SSA value because arts.edt is intentionally not IsolatedFromAbove.
  /// Run CSE at the EDT root while abstract EDTs are live so cleanup stays
  /// within the explicit dep/param boundary.
  pm.addNestedPass<arts::EdtOp>(createCSEPass());
}

static void addCanonicalizeAndEdtLocalCSE(PassManager &pm) {
  pm.addPass(polygeist::createPolygeistCanonicalizePass());
  addEdtLocalCSE(pm);
}

/// Normalize frontend storage and dependency shape before SDE conversion.
void buildSdeInputNormalizationPipeline(PassManager &pm) {
  OpPassManager &optPM = pm.nest<func::FuncOp>();
  /// Stage contract: normalize affine memory/control ops before the module pass
  /// runs so SdeInputNormalization only needs to reason about the
  /// memref+SCF form produced by the frontend/inliner pipeline.
  optPM.addPass(createLowerAffinePass());
  pm.addPass(createCSEPass());
  pm.addPass(sde::createSdeInputInlinerPass());
  pm.addPass(polygeist::createPolygeistCanonicalizePass());
  pm.addPass(sde::createScalarForwardingPass());
  pm.addPass(polygeist::createPolygeistCanonicalizePass());
  pm.addPass(sde::createSdeMemrefNormalizationPass());
  pm.addPass(sde::createSdeHandleDepsPass());
  pm.addPass(sde::createSdeDeadStateCleanupPass());
  pm.addPass(createCSEPass());
}

/// Initial cleanup and simplification passes.
void buildInitialCleanupPipeline(OpPassManager &optPM) {
  optPM.addPass(createLowerAffinePass());
  optPM.addPass(createCSEPass());
  optPM.addPass(polygeist::createCanonicalizeForPass());
}

/// OpenMP to SDE planning. Codelets are intentionally not lowered
/// here; SDE plans feed `sde-to-codir`, and CODIR then materializes ARTS.
void buildSdePlanningPipeline(PassManager &pm,
                              arts::AnalysisManager *AM = nullptr) {
  sde::SDECostModel *costModel = AM ? &AM->getCostModel() : nullptr;
  pm.addPass(sde::createConvertOpenMPToSdePass());
  // SDE pattern analysis first stamps approved memref/ND access facts. Dep
  // transforms then consume those SDE facts before effect passes make
  // scheduling decisions.
  pm.addPass(sde::createPatternAnalysisPass());
  pm.addPass(sde::createLoopInterchangePass());
  pm.addPass(sde::createTilingPass(costModel));
  pm.addPass(sde::createElementwiseFusionPass());
  pm.addPass(sde::createVectorizationPass(costModel));
  pm.addPass(sde::createScheduleRefinementPass(costModel));
  pm.addPass(sde::createChunkOptPass(costModel));
  pm.addPass(sde::createReductionStrategyPass(costModel));
  pm.addPass(sde::createDistributionPlanningPass(costModel));
  pm.addPass(sde::createIterationSpaceDecompositionPass());
  pm.addPass(sde::createBarrierEliminationPass(costModel));
  pm.addPass(sde::createVerifySdeCpsPlanPass());
  pm.addPass(sde::createMemoryUnitMaterializationPass());
}

/// SDE-to-CODIR materialization. This is the production codelet conversion:
/// SDE owns planning facts; CODIR owns isolated deps, params, and token-local
/// memref views.
void buildSdeToCodirPipeline(PassManager &pm) {
  pm.addPass(codir::createConvertSdeToCodirPass());
  pm.addPass(codir::createCodirCodeletOptPass());
  pm.addPass(codir::createVerifyCodirPass());
}

/// CODIR-to-ARTS materialization. Every ARTS EDT must come from a CODIR
/// codelet; any remaining SDE op is a boundary error.
void buildCodirToArtsPipeline(PassManager &pm) {
  pm.addPass(codir::createConvertCodirToArtsPass());
  pm.addPass(sde::createVerifySdeLoweredPass());
  pm.addPass(arts::createVerifyArtsObjectsOnlyPass());
  pm.addPass(arts::createDCEPass());
  addEdtLocalCSE(pm);
  pm.addPass(arts::createVerifyEdtCreatedPass());
}

/// EDT transformation passes.
void buildEdtTransformsPipeline(PassManager &pm, arts::AnalysisManager *AM) {
  pm.addPass(arts::createEdtStructuralOptPass(AM, false));
  pm.addPass(arts::createDCEPass());
  pm.addPass(createSymbolDCEPass());
  addEdtLocalCSE(pm);
  pm.addPass(arts::createEdtPtrRematerializationPass());
}

/// DB creation pass.
void buildCreateDbsPipeline(PassManager &pm, arts::AnalysisManager *AM) {
  pm.addPass(arts::createCreateDbsPass(AM));
  addCanonicalizeAndEdtLocalCSE(pm);
  pm.addPass(createSymbolDCEPass());
  pm.addPass(createMem2Reg());
  pm.addPass(polygeist::createPolygeistCanonicalizePass());
}

/// DB creation and optimization passes.
void buildDbOptPipeline(PassManager &pm, arts::AnalysisManager *AM) {
  pm.addPass(arts::createDbModeTighteningPass(AM));
  addCanonicalizeAndEdtLocalCSE(pm);
  pm.addPass(createMem2Reg());
}

/// Tighten DB modes and persist post-partition refinement contracts.
void buildPostDbRefinementPipeline(PassManager &pm, arts::AnalysisManager *AM,
                                   bool enableDistributedDb) {
  /// DbModeTighteningPass performs local DB cleanup after mode adjustment,
  /// which can expose new zero-dependency or degenerate EDTs before epoch
  /// shaping. Mode tightening must run before EDT transforms so affinity and
  /// reduction analysis see accurate writer/reader modes.
  pm.addPass(arts::createDbModeTighteningPass(AM));
  if (enableDistributedDb)
    pm.addPass(arts::createDbDistributedOwnershipPass(AM));
  pm.addPass(arts::createEdtTransformsPass(AM));
  /// Re-run DB transforms after EDT dependency pruning so cleanup-only
  /// acquires and now-unreachable DB roots are removed in the DB layer.
  pm.addPass(arts::createDbTransformsPass(AM));
  pm.addPass(arts::createContractValidationPass());
  pm.addPass(arts::createDbScratchEliminationPass());
  addCanonicalizeAndEdtLocalCSE(pm);
}

/// Apply late DB-aware loop cleanup and final stack/SSA simplification.
void buildLateConcurrencyCleanupPipeline(PassManager &pm) {
  pm.addNestedPass<func::FuncOp>(arts::createBlockLoopStripMiningPass());
  pm.addPass(arts::createHoistingPass());
  addCanonicalizeAndEdtLocalCSE(pm);
  /// TODO(PERF): EdtAllocaSinkingPass runs twice (here and pre-lowering).
  pm.addPass(arts::createEdtAllocaSinkingPass());
  pm.addPass(arts::createDCEPass());
  pm.addPass(createMem2Reg());
}

/// Epoch creation passes.
void buildEpochsPipeline(PassManager &pm, arts::AnalysisManager *AM) {
  pm.addPass(polygeist::createPolygeistCanonicalizePass());
  pm.addPass(arts::createCreateEpochsPass());
  pm.addPass(arts::createVerifyEpochCreatedPass());
  /// Run EpochOpt with structural + amortization optimizations on newly
  /// created epochs.
  pm.addPass(arts::createEpochOptPass(AM, /*amortization=*/true));
  pm.addPass(polygeist::createPolygeistCanonicalizePass());
}

/// Pre-lowering passes.
void buildPreLoweringPipeline(PassManager &pm) {
  /// TODO(PERF): EdtAllocaSinkingPass runs twice (late concurrency cleanup
  /// and here).
  pm.addPass(arts::createEdtAllocaSinkingPass());
  addCanonicalizeAndEdtLocalCSE(pm);
  pm.addPass(arts_rt::createDbLoweringPass(ArtsIdStride));
  addCanonicalizeAndEdtLocalCSE(pm);
  pm.addPass(arts_rt::createEdtLoweringPass(ArtsIdStride));
  addCanonicalizeAndCSE(pm);
  pm.addPass(arts_rt::createVerifyEdtLoweredPass());
  pm.addPass(createLoopInvariantCodeMotionPass());
  /// Hoist loop-invariant DB/dep pointer loads before scalar replacement;
  /// buildArtsRtToLLVMPipeline runs hoisting again after ARTS-RT-to-LLVM
  /// materializes new loads.
  pm.addPass(arts_rt::createDataPtrHoistingPass());
  addCanonicalizeAndCSE(pm);
  pm.addPass(arts_rt::createScalarReplacementPass());
  addCanonicalizeAndCSE(pm);
  pm.addPass(arts_rt::createEpochLoweringPass());
  addCanonicalizeAndCSE(pm);
  pm.addPass(arts_rt::createVerifyEpochLoweredPass());
  pm.addPass(arts_rt::createVerifyPreLoweredPass());
}

/// ARTS-RT to LLVM conversion passes.
void buildArtsRtToLLVMPipeline(PassManager &pm, bool debug,
                               bool distributedInitPerWorker,
                               const arts::RuntimeConfig *machine) {
  pm.addNestedPass<func::FuncOp>(createLowerAffinePass());
  pm.addPass(arts_rt::createConvertArtsRtToLLVMPass(
      debug, distributedInitPerWorker, machine));
  /// ConvertArtsRtToLLVM still consults lowering contracts for late dependency
  /// decisions (for example N-D stencil halo slices). Clean them up only after
  /// that conversion has consumed them.
  pm.addPass(arts_rt::createLoweringContractCleanupPass());
  pm.addPass(arts_rt::createGuidRangeCallOptPass());
  pm.addPass(arts_rt::createRuntimeCallOptPass());
  /// Hoist loop-invariant loads after ARTS-RT-to-LLVM lowering for
  /// vectorization/LICM.
  pm.addPass(arts_rt::createDataPtrHoistingPass());
  addCanonicalizeAndCSE(pm);
  pm.addPass(createMem2Reg());
  pm.addPass(polygeist::createPolygeistCanonicalizePass());
  pm.addPass(createControlFlowSinkPass());
  pm.addPass(polygeist::createPolygeistCanonicalizePass());
  pm.addPass(arts_rt::createVerifyDbLoweredPass());
  pm.addPass(arts_rt::createVerifyLoweredPass());
}

/// Additional optimizations (post-ARTS pipeline).
void buildAdditionalOptPipeline(OpPassManager &optPM) {
  optPM.addPass(polygeist::createPolygeistCanonicalizePass());
  optPM.addPass(createControlFlowSinkPass());
  optPM.addPass(polygeist::createPolygeistCanonicalizePass());
  optPM.addPass(createLoopInvariantCodeMotionPass());
  optPM.addPass(createCSEPass());
  optPM.addPass(polygeist::createPolygeistCanonicalizePass());
}

/// LLVM IR emission passes.
void buildLLVMIREmissionPipeline(PassManager &pm, bool convertOpenMP) {
  pm.addPass(createCSEPass());
  pm.addPass(polygeist::createPolygeistCanonicalizePass());
  pm.addPass(std::make_unique<MaterializeArtsFunctionPointersPass>());
  if (convertOpenMP)
    pm.addPass(createConvertOpenMPToLLVMPass());
  pm.addPass(arith::createArithExpandOpsPass());
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(std::make_unique<ResidualHostOpenMPMemrefCleanupPass>());
  pm.addPass(polygeist::createConvertPolygeistToLLVMPass());
  pm.addPass(createConvertIndexToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  pm.addPass(arts_rt::createAliasScopeGenPass());
  pm.addPass(arts_rt::createLoopVectorizationHintsPass());
  pm.addPass(polygeist::createPolygeistCanonicalizePass());
  pm.addPass(createCSEPass());
}

static bool isStageEnabledAlways(const StageExecutionContext &) { return true; }
static bool isStageEnabledWhenOptRequested(const StageExecutionContext &ctx) {
  return ctx.runAdditionalOpt;
}
static bool
isStageEnabledWhenEmitLLVMRequested(const StageExecutionContext &ctx) {
  return ctx.emitLLVM;
}

// --- Pipeline dependency declarations ---
static constexpr llvm::StringLiteral kDepSdeInputNormalization[] = {
    "sde-input-normalization"};
static constexpr llvm::StringLiteral kDepInitialCleanup[] = {"initial-cleanup"};
static constexpr llvm::StringLiteral kDepSdePlanning[] = {"sde-planning"};
static constexpr llvm::StringLiteral kDepSdeToCodir[] = {"sde-to-codir"};
static constexpr llvm::StringLiteral kDepCodirToArts[] = {"codir-to-arts"};
static constexpr llvm::StringLiteral kDepCreateDbs[] = {"create-dbs"};
static constexpr llvm::StringLiteral kDepPostDbRefinement[] = {
    "post-db-refinement"};
static constexpr llvm::StringLiteral kDepPreLowering[] = {
    "epochs", "late-concurrency-cleanup"};
static constexpr llvm::StringLiteral kDepArtsRtToLLVM[] = {"pre-lowering"};
static ArrayRef<StageDescriptor> getStageRegistry() {
  static const std::array<StageDescriptor, 15> kStageRegistry = {{
      {StageId::SdeInputNormalization, "sde-input-normalization", StageKind::Core, true, true,
       false, "Error when normalizing SDE input",
       kSdeInputNormalizationPasses,
       [](PassManager &pm, const StageExecutionContext &) {
         buildSdeInputNormalizationPipeline(pm);
       },
       isStageEnabledAlways,
       /*dependsOn=*/llvm::ArrayRef<llvm::StringLiteral>()},
      {StageId::InitialCleanup, "initial-cleanup", StageKind::Core, true, true,
       false, "Error simplifying the IR", kInitialCleanupPasses,
       [](PassManager &pm, const StageExecutionContext &) {
         OpPassManager &optPM = pm.nest<func::FuncOp>();
         buildInitialCleanupPipeline(optPM);
       },
       isStageEnabledAlways,
       /*dependsOn=*/kDepSdeInputNormalization},
      {StageId::SdePlanning, "sde-planning", StageKind::Core, true, true,
       false, "Error when converting OpenMP to SDE planning IR",
       kSdePlanningPasses,
       [](PassManager &pm, const StageExecutionContext &ctx) {
         buildSdePlanningPipeline(pm, ctx.analysisManager);
       },
       isStageEnabledAlways,
       /*dependsOn=*/kDepInitialCleanup},
      {StageId::SdeToCodir, "sde-to-codir", StageKind::Core, true, true,
       false, "Error when materializing SDE plans into CODIR codelets",
       kSdeToCodirPasses,
       [](PassManager &pm, const StageExecutionContext &) {
         buildSdeToCodirPipeline(pm);
       },
       isStageEnabledAlways,
       /*dependsOn=*/kDepSdePlanning},
      {StageId::CodirToArts, "codir-to-arts", StageKind::Core, true, true,
       false, "Error when materializing CODIR codelets into ARTS objects",
       kCodirToArtsPasses,
       [](PassManager &pm, const StageExecutionContext &) {
         buildCodirToArtsPipeline(pm);
       },
       isStageEnabledAlways,
       /*dependsOn=*/kDepSdeToCodir},
      {StageId::EdtTransforms, "edt-transforms", StageKind::Core, true, true,
       false, "Error when running EDT transformations", kEdtTransformsPasses,
       [](PassManager &pm, const StageExecutionContext &ctx) {
         buildEdtTransformsPipeline(pm, ctx.analysisManager);
       },
       isStageEnabledAlways,
       /*dependsOn=*/kDepCodirToArts},
      {StageId::CreateDbs, "create-dbs",
       StageKind::Core, true, true, false, "Error when creating DBs",
       kCreateDbsPasses,
       [](PassManager &pm, const StageExecutionContext &ctx) {
         buildCreateDbsPipeline(pm, ctx.analysisManager);
       },
       isStageEnabledAlways,
       /*dependsOn=*/kDepCodirToArts},
      {StageId::DbOpt, "db-opt",
       StageKind::Core, true, true, false, "Error when optimizing DBs",
       kDbOptPasses,
       [](PassManager &pm, const StageExecutionContext &ctx) {
         buildDbOptPipeline(pm, ctx.analysisManager);
       },
       isStageEnabledAlways,
       /*dependsOn=*/kDepCreateDbs},
      {StageId::PostDbRefinement, "post-db-refinement", StageKind::Core, true, true,
       false, "Error when refining post-partition DB contracts",
       kPostDbRefinementPasses,
       [](PassManager &pm, const StageExecutionContext &ctx) {
         buildPostDbRefinementPipeline(pm, ctx.analysisManager,
                                       DistributedDb);
       },
       isStageEnabledAlways,
       /*dependsOn=*/kDepCreateDbs},
      {StageId::LateConcurrencyCleanup, "late-concurrency-cleanup", StageKind::Core, true, true,
       false, "Error when running late concurrency cleanup",
       kLateConcurrencyCleanupPasses,
       [](PassManager &pm, const StageExecutionContext &) {
         buildLateConcurrencyCleanupPipeline(pm);
       },
       isStageEnabledAlways,
       /*dependsOn=*/kDepPostDbRefinement},
      {StageId::Epochs, "epochs",
       StageKind::Core, true, true, false,
       "Error when creating and optimizing epochs", kEpochsPasses,
       [](PassManager &pm, const StageExecutionContext &ctx) {
         buildEpochsPipeline(pm, ctx.analysisManager);
       },
       isStageEnabledAlways,
       /*dependsOn=*/kDepPostDbRefinement},
      {StageId::PreLowering, "pre-lowering", StageKind::Core, true, true, true,
       "Error when pre-lowering DBs, EDTs, and Epochs", kPreLoweringPasses,
       [](PassManager &pm, const StageExecutionContext &) {
         buildPreLoweringPipeline(pm);
       },
       isStageEnabledAlways,
       /*dependsOn=*/kDepPreLowering},
      {StageId::ArtsRtToLLVM, kArtsRtToLLVMToken, StageKind::Core, true, true,
       false, "Error when lowering ARTS-RT to LLVM", kArtsRtToLLVMPasses,
       [](PassManager &pm, const StageExecutionContext &ctx) {
         buildArtsRtToLLVMPipeline(pm, Debug, DistributedDb, ctx.machine);
       },
       isStageEnabledAlways,
       /*dependsOn=*/kDepArtsRtToLLVM},
      {StageId::PostO3Opt, kPostO3OptToken, StageKind::Epilogue, false, false,
       false, "Error when running classical optimizations", kPostO3OptPasses,
       [](PassManager &pm, const StageExecutionContext &) {
         OpPassManager &optPM = pm.nest<func::FuncOp>();
         buildAdditionalOptPipeline(optPM);
       },
       isStageEnabledWhenOptRequested,
       /*dependsOn=*/kDepArtsRtToLLVM},
      {StageId::LLVMIREmission, kLLVMIREmissionToken, StageKind::Epilogue, false, false,
       false, "Error when emitting LLVM IR", kLLVMIREmissionPasses,
       [](PassManager &pm, const StageExecutionContext &ctx) {
         buildLLVMIREmissionPipeline(pm, hasResidualOpenMP(ctx.module));
       },
       isStageEnabledWhenEmitLLVMRequested,
       /*dependsOn=*/kDepArtsRtToLLVM},
  }};
  return kStageRegistry;
}

/// Validate that no stage's dependsOn references a stage that comes after it
/// in the registry order. Call this during pipeline construction.
static bool validatePipelineDAG(ArrayRef<StageDescriptor> stages) {
  llvm::DenseMap<StringRef, unsigned> stageIndex;
  for (unsigned i = 0; i < stages.size(); ++i)
    stageIndex[stages[i].token] = i;

  for (unsigned i = 0; i < stages.size(); ++i) {
    for (StringRef dep : stages[i].dependsOn) {
      auto it = stageIndex.find(dep);
      if (it == stageIndex.end()) {
        llvm::errs() << "Pipeline error: stage '" << stages[i].token
                     << "' depends on unknown stage '" << dep << "'\n";
        return false;
      }
      if (it->second >= i) {
        llvm::errs() << "Pipeline error: stage '" << stages[i].token
                     << "' depends on '" << dep << "' which comes later\n";
        return false;
      }
    }
  }
  return true;
}

static FailureOr<StageId> resolveRequiredStageToken(StringRef token,
                                                    bool allowStartFrom) {
  if (const StageDescriptor *stage = findStageByToken(token)) {
    if (allowStartFrom && !stage->allowStartFrom)
      return failure();
    if (!allowStartFrom && !stage->allowPipelineStop)
      return failure();
    return stage->id;
  }
  return failure();
}

static FailureOr<std::optional<StageId>>
resolvePipelineStopToken(StringRef token) {
  if (token == kCompletePipelineToken)
    return std::optional<StageId>();
  FailureOr<StageId> resolved = resolveRequiredStageToken(token, false);
  if (failed(resolved))
    return failure();
  return std::optional<StageId>(*resolved);
}

static void emitAvailableStageTokens(bool startFrom) {
  ARTS_ERROR("Available " << (startFrom ? "start-from pipeline steps"
                                        : "pipeline steps")
                          << ":");
  for (const auto &stage : getStageRegistry()) {
    if (!(startFrom ? stage.allowStartFrom : stage.allowPipelineStop))
      continue;
    ARTS_ERROR("- " << stage.token);
  }
  if (!startFrom)
    ARTS_ERROR("- " << kCompletePipelineToken);
}

static void emitUnknownStageTokenError(StringRef token, bool startFrom) {
  ARTS_ERROR(
      "Unknown " << (startFrom ? "start-from pipeline step" : "pipeline step")
                 << ": '" << token << "'");
  emitAvailableStageTokens(startFrom);
}

/// Hooks invoked around each stage for diagnostics or custom logging.
struct PipelineHooks {
  std::function<void(StageId)> beforeStep;
  std::function<void(StageId, LogicalResult)> afterStep;
};

/// Configure the pass manager with the optimization passes.
LogicalResult
buildPassManager(ModuleOp module, MLIRContext &context,
                 std::optional<StageId> stopAt,
                 StageId startFrom = StageId::SdeInputNormalization,
                 std::unique_ptr<arts::AnalysisManager> *outAM = nullptr,
                 PipelineHooks *hooks = nullptr) {
  assert(validatePipelineDAG(getStageRegistry()) &&
         "Pipeline dependency DAG is invalid");

  int startIndex = stageIndex(startFrom, StageKind::Core);
  int stopIndex = stopAt ? stageIndex(*stopAt, StageKind::Core)
                         : stageIndex(StageId::ArtsRtToLLVM, StageKind::Core);
  if (startIndex < 0 || stopIndex < 0) {
    ARTS_ERROR(
        "Invalid pipeline selection: --start-from="
        << stageName(startFrom) << ", --pipeline="
        << (stopAt ? stageName(*stopAt) : StringRef(kCompletePipelineToken)));
    return failure();
  }
  if (startIndex > stopIndex) {
    ARTS_ERROR(
        "Invalid pipeline range: --start-from="
        << stageName(startFrom) << " is after --pipeline="
        << (stopAt ? stageName(*stopAt) : StringRef(kCompletePipelineToken)));
    return failure();
  }

  /// Create module-level analysis manager for caching across functions
  std::unique_ptr<arts::AnalysisManager> AM =
      std::make_unique<arts::AnalysisManager>(module, ArtsConfig);

  auto &machine = AM->getRuntimeConfig();
  if (!machine.hasConfigFile() || !machine.isValid()) {
    ARTS_ERROR("Invalid ARTS configuration. Provide a valid --arts-config path "
               "or place a valid arts.cfg in the working directory.");
    return failure();
  }

  /// Embed config file contents into the module so generated binaries are
  /// self-contained — no external config file needed at runtime.
  if (machine.hasConfigFile() && !machine.getConfigPath().empty()) {
    auto configContents = llvm::MemoryBuffer::getFile(machine.getConfigPath());
    if (configContents)
      arts::setRuntimeConfigData(module, (*configContents)->getBuffer());
    else
      arts::setRuntimeConfigPath(module, machine.getConfigPath());
  }
  arts::setRuntimeTotalWorkers(module, machine.getRuntimeTotalWorkers());
  arts::setRuntimeTotalNodes(module, machine.getNodeCount());
  arts::setRuntimeStaticWorkers(module, RuntimeStaticWorkers);

  if (machine.getNodeCount() > 1 && !DistributedDb)
    ARTS_WARN("Multi-node execution without --distributed-db: all DBs will "
              "be created on their origin node");
  /// Create shared timing data for pass instrumentation.
  arts::PassTimingData timingData;
  arts::PassTimingData *timingDataPtr = PassTiming ? &timingData : nullptr;

  auto runStage = [&](const StageDescriptor &stage,
                      bool stopAfterStage) -> LogicalResult {
    if (hooks && hooks->beforeStep)
      hooks->beforeStep(stage.id);

    if (stage.captureDiagnosticsBefore && outAM)
      AM->captureDiagnostics();

    PassManager pm(&context);
    if (failed(configurePassManager(pm, timingDataPtr))) {
      ARTS_ERROR("Error configuring pass manager for pipeline step "
                 << stage.token);
      return failure();
    }
    StageExecutionContext stageContext{
        module, context, AM.get(), &machine, stopAfterStage, Opt, EmitLLVM};
    stage.build(pm, stageContext);
    auto result = pm.run(module);

    if (hooks && hooks->afterStep)
      hooks->afterStep(stage.id, result);

    if (failed(result)) {
      ARTS_ERROR(stage.errorMessage);
      module->dump();
      return failure();
    }
    return success();
  };

  auto releaseAnalysisManager = [&]() {
    if (outAM)
      *outAM = std::move(AM);
  };

  for (const auto &stage : getStageRegistry()) {
    if (stage.kind != StageKind::Core)
      continue;
    int currentIndex = stageIndex(stage.id, StageKind::Core);
    if (currentIndex < startIndex)
      continue;
    if (!stage.enabled(StageExecutionContext{module, context, AM.get(),
                                             &machine, false, Opt, EmitLLVM}))
      continue;
    bool stopHere = stopAt && *stopAt == stage.id;
    if (failed(runStage(stage, stopHere)))
      return failure();
    if (stopHere) {
      if (PassTiming)
        timingData.printTimingReport(llvm::errs());
      releaseAnalysisManager();
      return success();
    }
  }

  for (const auto &stage : getStageRegistry()) {
    if (stage.kind != StageKind::Epilogue)
      continue;
    StageExecutionContext stageContext{module, context, AM.get(), &machine,
                                       false,  Opt,     EmitLLVM};
    if (!stage.enabled(stageContext))
      continue;
    if (failed(runStage(stage, /*stopAfterStage=*/false)))
      return failure();
  }

  /// Print pass timing report if enabled.
  if (PassTiming)
    timingData.printTimingReport(llvm::errs());
  if (!PassTimingOutput.empty()) {
    std::error_code EC;
    llvm::raw_fd_ostream timingFile(PassTimingOutput, EC);
    if (!EC) {
      timingData.exportTimingJson(timingFile);
      timingFile.close();
    } else {
      ARTS_WARN("Could not open pass timing output file: " << PassTimingOutput);
    }
  }

  /// Return analysis manager if requested
  releaseAnalysisManager();

  return success();
}

///===----------------------------------------------------------------------===///
/// Main Function
///===----------------------------------------------------------------------===///
int main(int argc, char **argv) {
  InitLLVM y(argc, argv);
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "MLIR Optimization Driver\n");
  std::string effectiveArtsDebug = ArtsDebug;
  if (Diagnose) {
    if (!effectiveArtsDebug.empty())
      effectiveArtsDebug += ",";
    effectiveArtsDebug += "compile";
  }
  configureArtsDebugChannels(effectiveArtsDebug);

  if (PrintPipelineManifestJSON) {
    printPipelineManifestAsJSON(llvm::outs());
    return 0;
  }

  FailureOr<StageId> resolvedStartFrom =
      resolveRequiredStageToken(StartFrom.getValue(), /*allowStartFrom=*/true);
  if (failed(resolvedStartFrom)) {
    emitUnknownStageTokenError(StartFrom, /*startFrom=*/true);
    return 1;
  }

  std::string effectivePipelineToken = Pipeline;

  FailureOr<std::optional<StageId>> resolvedStopAt =
      resolvePipelineStopToken(effectivePipelineToken);
  if (failed(resolvedStopAt)) {
    emitUnknownStageTokenError(effectivePipelineToken, /*startFrom=*/false);
    return 1;
  }

  /// Set up the dialect registry and MLIR context.
  DialectRegistry registry;
  registerDialects(registry);
  MLIRContext context(registry);
  initializeContext(context);

  /// Open the input file.
  auto file = openInputFile(InputFilename);
  if (!file) {
    ARTS_ERROR("Could not open input file: " << InputFilename);
    return 1;
  }

  /// Parse the input module.
  auto module = parseSourceString<ModuleOp>(file->getBuffer(), &context);
  if (!module) {
    ARTS_ERROR("Could not parse input file");
    return 1;
  }

  if (!CustomPassPipeline.empty()) {
    PassManager pm(&context);
    if (failed(configurePassManager(pm))) {
      ARTS_ERROR("Error configuring pass manager for --pass-pipeline");
      return 1;
    }
    StringRef passPipeline = CustomPassPipeline;
    std::string unwrappedPassPipeline;
    if (passPipeline.consume_front("builtin.module(") &&
        passPipeline.consume_back(")")) {
      unwrappedPassPipeline = passPipeline.str();
      passPipeline = unwrappedPassPipeline;
    } else {
      passPipeline = CustomPassPipeline;
    }
    if (failed(parsePassPipeline(passPipeline, pm))) {
      ARTS_ERROR("Could not parse --pass-pipeline: " << CustomPassPipeline);
      return 1;
    }
    if (failed(pm.run(module.get()))) {
      ARTS_ERROR("Error running --pass-pipeline: " << CustomPassPipeline);
      module->dump();
      return 1;
    }

    auto output = openOutputFile(OutputFilename);
    if (!output) {
      ARTS_ERROR("Could not open output file: " << OutputFilename);
      return 1;
    }
    module->print(output->os());
    output->keep();
    return 0;
  }

  /// Set up optional pipeline hooks for diagnostics.
  PipelineHooks hooks;
  PipelineHooks *hooksPtr = nullptr;
  if (Diagnose) {
    hooks.afterStep = [](StageId stage, LogicalResult result) {
      ARTS_INFO("Pipeline " << stageName(stage)
                            << (succeeded(result) ? " completed" : " FAILED"));
    };
    hooksPtr = &hooks;
  }

  /// Run the pass pipeline once.
  std::unique_ptr<arts::AnalysisManager> AM;
  if (failed(buildPassManager(module.get(), context, *resolvedStopAt,
                              *resolvedStartFrom, Diagnose ? &AM : nullptr,
                              hooksPtr))) {
    return 1;
  }

  /// Export diagnostics if requested
  if (Diagnose && AM) {
    bool includeAnalysis = shouldExportDetailedDiagnose(*resolvedStopAt) &&
                           AM->hasCapturedDiagnostics();
    if (!DiagnoseOutput.empty()) {
      /// Export to file
      std::error_code EC;
      llvm::raw_fd_ostream outputFile(DiagnoseOutput, EC);
      if (EC) {
        ARTS_ERROR(
            "Could not open diagnostics output file: " << DiagnoseOutput);
        return 1;
      }
      AM->exportToJson(outputFile, includeAnalysis);
      outputFile.close();
    } else {
      /// Export to stdout
      AM->exportToJson(llvm::outs(), includeAnalysis);
    }
  }

  /// Translate the optimized module to LLVM IR and write output.
  if (EmitLLVM) {
    bool hasHostOpenMP = hasResidualOpenMP(module.get());
    if (hasHostOpenMP)
      foldResidualHostOpenMPMemrefPointerCasts(module.get());
    ensureRuntimeConfigDataVisibleForValidation(module.get());
    markHostOpenMPBenchmarkMode(module.get());
    LLVMContext llvmContext;
    auto llvmModule = translateModuleToLLVMIR(module.get(), llvmContext);
    if (!llvmModule) {
      module->dump();
      ARTS_ERROR("Failed to emit LLVM IR");
      return -1;
    }
    if (hasHostOpenMP)
      bypassArtsRuntimeForHostOpenMP(*llvmModule);
    std::string llvmIR;
    raw_string_ostream llvmStream(llvmIR);
    llvmModule->print(llvmStream, nullptr);

    auto output = openOutputFile(OutputFilename);
    if (!output) {
      ARTS_ERROR("Could not open output file: " << OutputFilename);
      return 1;
    }
    output->os() << llvmStream.str();
    output->keep();
  } else {
    /// Otherwise, print the final MLIR module.
    auto output = openOutputFile(OutputFilename);
    if (!output) {
      ARTS_ERROR("Could not open output file: " << OutputFilename);
      return 1;
    }
    module->print(output->os());
    output->keep();
  }

  return 0;
}
