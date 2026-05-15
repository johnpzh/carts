add_mlir_dialect_library(MLIRCartsSde
  SdeDialect.cpp
  SdeOps.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/carts/dialect/sde/IR

  DEPENDS
    MLIRSdeOpsIncGen

  LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRIR
    MLIRMemRefDialect
)
