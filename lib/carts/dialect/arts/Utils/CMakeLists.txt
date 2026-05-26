add_mlir_dialect_library(MLIRCartsArtsUtils
  DbLayoutPlanUtils.cpp
  DbUtils.cpp
  EdtUtils.cpp
  LocationMetadata.cpp
  LoopStructureUtils.cpp
  LoweringContractUtils.cpp
  RuntimeConfig.cpp
  ValueAnalysisUtils.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/carts/dialect/arts/Utils

  DEPENDS
    MLIRArtsOpsIncGen
    MLIRSdeOpsIncGen

  LINK_LIBS PUBLIC
    MLIRAffineDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRLLVMDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRSideEffectInterfaces
    MLIRTransformUtils
    MLIRCartsArts
    MLIRCartsSde
)

target_link_libraries(MLIRCartsArtsUtils PRIVATE ${MLIR_POLYGEIST_LIB})
