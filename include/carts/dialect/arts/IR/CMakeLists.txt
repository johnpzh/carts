set(LLVM_TARGET_DEFINITIONS ArtsOps.td)
mlir_tablegen(ArtsOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(ArtsOpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(ArtsOpsAttributes.h.inc -gen-attrdef-decls
              -attrdefs-dialect=arts)
mlir_tablegen(ArtsOpsAttributes.cpp.inc -gen-attrdef-defs
              -attrdefs-dialect=arts)
add_mlir_dialect(ArtsOps arts)
