#ifndef ARTS_UTILS_PLANCONTRACT_H
#define ARTS_UTILS_PLANCONTRACT_H

#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <optional>

namespace mlir {
namespace arts {

/// Shared vocabulary for semantic execution plans stamped by SDE and consumed
/// by Core orchestration passes.
enum class KernelFamily : uint8_t {
  Uniform,
  Stencil,
  Wavefront,
  ReductionMixed,
  TimestepChain,
  Unknown
};

inline llvm::StringRef kernelFamilyToString(KernelFamily family) {
  switch (family) {
  case KernelFamily::Uniform:
    return "uniform";
  case KernelFamily::Stencil:
    return "stencil";
  case KernelFamily::Wavefront:
    return "wavefront";
  case KernelFamily::ReductionMixed:
    return "reduction_mixed";
  case KernelFamily::TimestepChain:
    return "timestep_chain";
  case KernelFamily::Unknown:
    return "unknown";
  }
  return "unknown";
}

inline std::optional<KernelFamily> parseKernelFamily(llvm::StringRef str) {
  if (str == "uniform")
    return KernelFamily::Uniform;
  if (str == "stencil")
    return KernelFamily::Stencil;
  if (str == "wavefront")
    return KernelFamily::Wavefront;
  if (str == "reduction_mixed")
    return KernelFamily::ReductionMixed;
  if (str == "timestep_chain")
    return KernelFamily::TimestepChain;
  if (str == "unknown")
    return KernelFamily::Unknown;
  return std::nullopt;
}

enum class RepetitionStructure : uint8_t {
  None,
  PairStep,
  KStep,
  FullTimestep
};

inline llvm::StringRef
repetitionStructureToString(RepetitionStructure repetition) {
  switch (repetition) {
  case RepetitionStructure::None:
    return "none";
  case RepetitionStructure::PairStep:
    return "pair_step";
  case RepetitionStructure::KStep:
    return "k_step";
  case RepetitionStructure::FullTimestep:
    return "full_timestep";
  }
  return "none";
}

inline std::optional<RepetitionStructure>
parseRepetitionStructure(llvm::StringRef str) {
  if (str == "none")
    return RepetitionStructure::None;
  if (str == "pair_step")
    return RepetitionStructure::PairStep;
  if (str == "k_step")
    return RepetitionStructure::KStep;
  if (str == "full_timestep")
    return RepetitionStructure::FullTimestep;
  return std::nullopt;
}

} // namespace arts
} // namespace mlir

#endif // ARTS_UTILS_PLANCONTRACT_H
