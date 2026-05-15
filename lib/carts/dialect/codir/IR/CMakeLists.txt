add_mlir_dialect_library(MLIRCartsCodir
  CodirDialect.cpp
  CodirOps.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/carts/dialect/codir/IR

  DEPENDS
    MLIRCodirOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSideEffectInterfaces
)
