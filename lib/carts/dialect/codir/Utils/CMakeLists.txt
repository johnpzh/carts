add_mlir_dialect_library(MLIRCartsCodirUtils
  CodeletABIUtils.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/carts/dialect/codir/Utils

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRMemRefDialect
)
