set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(OpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(OpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(OpsAttributes.h.inc -gen-attrdef-decls
              -attrdefs-dialect=arts)
mlir_tablegen(OpsAttributes.cpp.inc -gen-attrdef-defs
              -attrdefs-dialect=arts)
add_mlir_dialect(Ops arts)
