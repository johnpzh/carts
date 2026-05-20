add_mlir_dialect_library(MLIRCartsCodirTransforms
  CodirCodeletOpt.cpp
  ReductionPlanning.cpp
  StoragePlanning.cpp
  VerifyCodir.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/carts/dialect/codir/Transforms
    ${PROJECT_SOURCE_DIR}/include/carts/dialect/codir/Utils

  DEPENDS
    MLIRCartsCodirPassIncGen
    MLIRCodirOpsIncGen

  LINK_LIBS PUBLIC
    MLIRCartsCodir
    MLIRCartsCodirUtils
    MLIRIR
    MLIRPass
    MLIRSideEffectInterfaces
)
