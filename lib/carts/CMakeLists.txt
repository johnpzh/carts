add_mlir_dialect_library(MLIRArts
  dialect/arts/IR/Dialect.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/carts

  DEPENDS
    MLIROpsIncGen
    MLIRRtOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRMemRefDialect
    MLIRLLVMDialect
    MLIROpenMPDialect
    MLIRAffineDialect
    MLIRLoopLikeInterface
    MLIRSupport
    MLIRSCFTransforms
)
add_subdirectory(dialect)
add_subdirectory(utils)
add_subdirectory(dialect/arts/Transforms)
add_subdirectory(dialect/arts/Conversion)
add_subdirectory(dialect/arts/Analysis)
add_subdirectory(dialect/arts/Utils)
add_subdirectory(passes)
