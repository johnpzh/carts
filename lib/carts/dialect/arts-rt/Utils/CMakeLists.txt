add_mlir_dialect_library(MLIRCartsArtsRtUtils
  IdRegistry.cpp
  RtDbUtils.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/carts/dialect/arts-rt/Utils

  DEPENDS
    MLIROpsIncGen
    MLIRRtOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRLLVMDialect
    MLIRMemRefDialect
    MLIRCartsArts
    MLIRCartsArtsRt
    MLIRCartsArtsUtils
)

target_link_libraries(MLIRCartsArtsRtUtils PRIVATE ${MLIR_POLYGEIST_LIB})
