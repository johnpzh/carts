add_mlir_dialect_library(MLIRCartsSdeToCodirConversion
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
    MLIRCartsSde
    MLIRCartsCodir
    MLIRCartsCodirUtils
    MLIRIR
    MLIRMemRefDialect
    MLIRPass
    MLIRSCFDialect
)

target_link_libraries(MLIRCartsSdeToCodirConversion
  PRIVATE ${MLIR_POLYGEIST_LIB})

add_mlir_dialect_library(MLIRCartsCodirToArtsConversion
  CodirToArts/CodirToArts.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/carts/dialect/codir/Transforms
    ${PROJECT_SOURCE_DIR}/include/carts/dialect/codir/Utils
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/CodirToArts
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

target_link_libraries(MLIRCartsCodirToArtsConversion
  PRIVATE ${MLIR_POLYGEIST_LIB})
