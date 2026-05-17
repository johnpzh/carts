add_mlir_dialect_library(MLIRCartsSdeUtils
  IterationSizingUtils.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/carts/dialect/sde/Utils

  DEPENDS
    MLIRSdeOpsIncGen

  LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRIR
    MLIRCartsSde
)
