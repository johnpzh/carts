###===---------------------------------------------------------------===###
### Per-dialect pass libraries. Consumers link the dialect libraries they use
### instead of a project-wide aggregate.
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
  MLIRArtsOpsIncGen
  MLIRCartsArtsPassIncGen
  MLIRCartsSdePassIncGen
  MLIRArtsRtOpsIncGen
  MLIRSdeOpsIncGen

LINK_LIBS PUBLIC
  ${_carts_common_link_libs}
  MLIRCartsSdeUtils
)

###===---------------------------------------------------------------===###
### MLIRCartsArtsTransforms — ARTS analysis, conversion, transforms, utils
###===---------------------------------------------------------------===###
add_mlir_dialect_library(MLIRCartsArtsTransforms
  PARTIAL_SOURCES_INTENDED
  ${CARTS_TRANSFORMS_SOURCE_FILES}
  ${CARTS_ANALYSIS_SOURCE_FILES}
  ${CARTS_UTILS_SOURCE_FILES}

DEPENDS
  MLIRArtsOpsIncGen
  MLIRCartsArtsPassIncGen
  MLIRCartsArtsRtPassIncGen
  MLIRCartsSdePassIncGen
  MLIRArtsRtOpsIncGen
  MLIRSdeOpsIncGen

LINK_LIBS PUBLIC
  ${_carts_common_link_libs}
  MLIRCartsArtsUtils
)

###===---------------------------------------------------------------===###
### MLIRCartsArtsRtTransforms — ARTS-RT conversion, transforms, utils
###===---------------------------------------------------------------===###
add_mlir_dialect_library(MLIRCartsArtsRtTransforms
  PARTIAL_SOURCES_INTENDED
  ${CARTS_RT_TO_LLVM_SOURCE_FILES}
  ${CARTS_RT_TRANSFORMS_SOURCE_FILES}

DEPENDS
  MLIRArtsOpsIncGen
  MLIRCartsArtsRtPassIncGen
  MLIRArtsRtOpsIncGen

LINK_LIBS PUBLIC
  ${_carts_common_link_libs}
  MLIRCartsArtsRtUtils
)

# All three per-dialect libraries need filesystem support on older toolchains
# and the Polygeist private dep.
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
