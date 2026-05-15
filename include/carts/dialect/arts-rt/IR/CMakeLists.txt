set(LLVM_TARGET_DEFINITIONS RtOps.td)
mlir_tablegen(RtOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(RtOpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(RtOpsAttributes.h.inc -gen-attrdef-decls
              -attrdefs-dialect=arts_rt)
mlir_tablegen(RtOpsAttributes.cpp.inc -gen-attrdef-defs
              -attrdefs-dialect=arts_rt)
add_mlir_dialect(RtOps arts_rt)
