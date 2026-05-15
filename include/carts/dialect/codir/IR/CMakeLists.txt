set(LLVM_TARGET_DEFINITIONS CodirOps.td)
mlir_tablegen(CodirOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(CodirOpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(CodirOpsAttributes.h.inc -gen-attrdef-decls
              -attrdefs-dialect=codir)
mlir_tablegen(CodirOpsAttributes.cpp.inc -gen-attrdef-defs
              -attrdefs-dialect=codir)
add_mlir_dialect(CodirOps codir)
