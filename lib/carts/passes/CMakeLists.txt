###===---------------------------------------------------------------===###
### Per-dialect pass libraries (split from the former monolithic
### MLIRCartsTransforms umbrella). MLIRCartsTransforms remains as a thin
### umbrella that aggregates the per-dialect libraries.
###===---------------------------------------------------------------===###

set(_carts_common_link_libs
  MLIRAffineDialect
  MLIRAffineUtils
  MLIRArithDialect
  MLIRArithToLLVM
  MLIRAsyncDialect
  MLIRBufferizationDialect
  MLIRControlFlowToLLVM
  MLIRFuncDialect
  MLIRFuncToLLVM
  MLIRFuncTransforms
  MLIRLLVMDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMathDialect
  MLIRMathToLLVM
  MLIRMemRefDialect
  MLIRMemRefToLLVM
  MLIRNVVMDialect
  MLIROpenMPToLLVM
  MLIRPass
  MLIRSCFToControlFlow
  MLIRSideEffectInterfaces
  MLIRTargetLLVMIRImport
  MLIRTensorDialect
  MLIRTransformUtils
  MLIRCartsArts
  MLIRCartsArtsRt
  MLIRCartsSde
)

###===---------------------------------------------------------------===###
### MLIRCartsSdeTransforms — SDE analysis, conversion, transforms, verify
###===---------------------------------------------------------------===###
add_mlir_dialect_library(MLIRCartsSdeTransforms
  PARTIAL_SOURCES_INTENDED
  ${CARTS_SDE_ANALYSIS_SOURCE_FILES}
  ${CARTS_SDE_CONVERSION_SOURCE_FILES}
  ${CARTS_SDE_TRANSFORMS_SOURCE_FILES}
  ${CARTS_SDE_VERIFY_SOURCE_FILES}

DEPENDS
  MLIROpsIncGen
  MLIRCartsArtsPassIncGen
  MLIRCartsSdePassIncGen
  MLIRRtOpsIncGen
  MLIRSdeOpsIncGen

LINK_LIBS PUBLIC
  ${_carts_common_link_libs}
)

###===---------------------------------------------------------------===###
### MLIRCartsArtsTransforms — ARTS analysis, conversion, transforms, utils
###===---------------------------------------------------------------===###
add_mlir_dialect_library(MLIRCartsArtsTransforms
  PARTIAL_SOURCES_INTENDED
  ${CARTS_TRANSFORMS_SOURCE_FILES}
  ${CARTS_ARTS_UTILS_SOURCE_FILES}
  ${CARTS_ANALYSIS_SOURCE_FILES}
  ${CARTS_UTILS_SOURCE_FILES}

DEPENDS
  MLIROpsIncGen
  MLIRCartsArtsPassIncGen
  MLIRCartsArtsRtPassIncGen
  MLIRCartsSdePassIncGen
  MLIRRtOpsIncGen
  MLIRSdeOpsIncGen

LINK_LIBS PUBLIC
  ${_carts_common_link_libs}
)

###===---------------------------------------------------------------===###
### MLIRCartsArtsRtTransforms — ARTS-RT conversion, transforms, utils
###===---------------------------------------------------------------===###
add_mlir_dialect_library(MLIRCartsArtsRtTransforms
  PARTIAL_SOURCES_INTENDED
  ${CARTS_RT_TO_LLVM_SOURCE_FILES}
  ${CARTS_RT_TRANSFORMS_SOURCE_FILES}
  ${CARTS_ARTS_RT_UTILS_SOURCE_FILES}

DEPENDS
  MLIROpsIncGen
  MLIRCartsArtsRtPassIncGen
  MLIRRtOpsIncGen

LINK_LIBS PUBLIC
  ${_carts_common_link_libs}
)

###===---------------------------------------------------------------===###
### MLIRCartsTransforms — thin umbrella that pulls in all dialect libs.
### Existing consumers link MLIRCartsTransforms; new consumers may link a
### specific per-dialect library directly.
###===---------------------------------------------------------------===###
add_mlir_dialect_library(MLIRCartsTransforms
  PARTIAL_SOURCES_INTENDED
  Umbrella.cpp

LINK_LIBS PUBLIC
  MLIRCartsSdeTransforms
  MLIRCartsArtsTransforms
  MLIRCartsArtsRtTransforms
)

# All three per-dialect libraries (and the umbrella) need filesystem support
# on older toolchains, and the Polygeist private dep.
foreach(_lib MLIRCartsSdeTransforms MLIRCartsArtsTransforms MLIRCartsArtsRtTransforms)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
      target_link_libraries(${_lib} PUBLIC stdc++fs)
    endif()
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7)
      target_link_libraries(${_lib} PUBLIC c++experimental)
    elseif(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
      target_link_libraries(${_lib} PUBLIC c++fs)
    endif()
  endif()
  target_link_libraries(${_lib} PRIVATE ${MLIR_POLYGEIST_LIB})
  target_compile_definitions(obj.${_lib}
    PRIVATE
    CARTS_PGO_DEFAULT_DATA_DIR="${CARTS_PGO_DEFAULT_DATA_DIR}"
    CARTS_PGO_ALTERNATIVE_ENV_VAR="${CARTS_PGO_ALTERNATIVE_ENV_VAR}"
    CARTS_PGO_DATA_DIR_ENV_VAR="${CARTS_PGO_DATA_DIR_ENV_VAR}"
  )
endforeach()
