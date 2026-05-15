add_mlir_dialect_library(MLIRArtsTransforms
  ${CARTS_SDE_ANALYSIS_SOURCE_FILES}
  ${CARTS_SDE_CONVERSION_SOURCE_FILES}
  ${CARTS_SDE_TRANSFORMS_SOURCE_FILES}
  ${CARTS_SDE_VERIFY_SOURCE_FILES}
  ${CARTS_CORE_CONVERSION_SOURCE_FILES}
  ${CARTS_TRANSFORMS_SOURCE_FILES}
  ${CARTS_ARTS_UTILS_SOURCE_FILES}
  ${CARTS_RT_TO_LLVM_SOURCE_FILES}
  ${CARTS_RT_TRANSFORMS_SOURCE_FILES}
  ${CARTS_ARTS_RT_UTILS_SOURCE_FILES}
  ${CARTS_UTILS_SOURCE_FILES}
  ${CARTS_ANALYSIS_SOURCE_FILES}

DEPENDS
  MLIROpsIncGen
  MLIRArtsPassIncGen
  MLIRArtsRtPassIncGen
  MLIRCartsSdePassIncGen
  MLIRRtOpsIncGen
  MLIRSdeOpsIncGen

LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRAffineUtils
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRLLVMDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMathDialect
  MLIRMathToLLVM
  MLIRMemRefDialect
  MLIRNVVMDialect
  MLIRPass
  MLIRArts
  MLIRArtsRt
  MLIRCartsSde
  MLIRSideEffectInterfaces
  MLIRSCFToControlFlow
  MLIRTargetLLVMIRImport
  MLIRTensorDialect
  MLIRTransformUtils
  MLIRControlFlowToLLVM
  MLIRMemRefToLLVM
  MLIRFuncToLLVM
  MLIRArithToLLVM
  MLIROpenMPToLLVM
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(MLIRArtsTransforms PUBLIC stdc++fs)
  endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7)
    target_link_libraries(MLIRArtsTransforms PUBLIC c++experimental)
  elseif(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(MLIRArtsTransforms PUBLIC c++fs)
  endif()
endif()

# Find Polygeist lib
target_link_libraries(MLIRArtsTransforms PRIVATE ${MLIR_POLYGEIST_LIB})

target_compile_definitions(obj.MLIRArtsTransforms
  PRIVATE
  CARTS_PGO_DEFAULT_DATA_DIR="${CARTS_PGO_DEFAULT_DATA_DIR}"
  CARTS_PGO_ALTERNATIVE_ENV_VAR="${CARTS_PGO_ALTERNATIVE_ENV_VAR}"
  CARTS_PGO_DATA_DIR_ENV_VAR="${CARTS_PGO_DATA_DIR_ENV_VAR}"
)
