set(LLVM_TARGET_DEFINITIONS SdeOps.td)
mlir_tablegen(SdeOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(SdeOpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(SdeOpsAttributes.h.inc -gen-attrdef-decls
              -attrdefs-dialect=sde)
mlir_tablegen(SdeOpsAttributes.cpp.inc -gen-attrdef-defs
              -attrdefs-dialect=sde)
mlir_tablegen(SdeOpsTypes.h.inc -gen-typedef-decls
              -typedefs-dialect=sde)
mlir_tablegen(SdeOpsTypes.cpp.inc -gen-typedef-defs
              -typedefs-dialect=sde)
add_mlir_dialect(SdeOps sde)
