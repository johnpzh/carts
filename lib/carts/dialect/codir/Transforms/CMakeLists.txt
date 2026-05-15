add_mlir_dialect_library(MLIRCartsCodirTransforms
  CodirBoundaryPlaceholders.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/carts/dialect/codir/Transforms

  DEPENDS
    MLIRCartsCodirPassIncGen
    MLIRCodirOpsIncGen
    MLIROpsIncGen
    MLIRSdeOpsIncGen

  LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRArts
    MLIRCartsSde
    MLIRCartsCodir
    MLIRIR
    MLIRMemRefDialect
    MLIRPass
)
