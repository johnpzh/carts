// RUN: %carts-compile %s --pass-pipeline='builtin.module(verify-codir)'

module {
  llvm.mlir.global internal constant @str0("source printf\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32

  func.func @codir_allows_source_llvm_ir() {
    codir.codelet {
      %addr = llvm.mlir.addressof @str0 : !llvm.ptr
      %ptr = llvm.getelementptr %addr[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<15 x i8>
      %unused = llvm.call @printf(%ptr) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      codir.yield
    }
    return
  }
}
