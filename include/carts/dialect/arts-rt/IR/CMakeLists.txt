set(LLVM_TARGET_DEFINITIONS ArtsRtOps.td)
mlir_tablegen(ArtsRtOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(ArtsRtOpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(ArtsRtOpsAttributes.h.inc -gen-attrdef-decls
              -attrdefs-dialect=arts_rt)
mlir_tablegen(ArtsRtOpsAttributes.cpp.inc -gen-attrdef-defs
              -attrdefs-dialect=arts_rt)
add_mlir_dialect(ArtsRtOps arts_rt)
