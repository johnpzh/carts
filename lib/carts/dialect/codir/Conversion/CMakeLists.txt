add_mlir_dialect_library(MLIRCartsCodirConversion
  CodirToArts/CodirToArts.cpp
  SdeToCodir/SdeToCodir.cpp
  SdeToCodir/TaskDepSliceUtils.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/carts/dialect/codir/Transforms
    ${PROJECT_SOURCE_DIR}/include/carts/dialect/codir/Utils
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/SdeToCodir

  DEPENDS
    MLIRCartsCodirConversionPassIncGen
    MLIRCodirOpsIncGen
    MLIROpsIncGen
    MLIRSdeOpsIncGen

  LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRCartsArts
    MLIRCartsSde
    MLIRCartsCodir
    MLIRCartsCodirUtils
    MLIRIR
    MLIRMemRefDialect
    MLIRPass
    MLIRSCFDialect
)

target_link_libraries(MLIRCartsCodirConversion PRIVATE ${MLIR_POLYGEIST_LIB})
