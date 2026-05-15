add_mlir_dialect_library(MLIRCartsArtsRt
  RtDialect.cpp
  RtOps.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/carts/dialect/arts-rt/IR

  DEPENDS
    MLIRRtOpsIncGen

  LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRIR
    MLIRMemRefDialect
)
