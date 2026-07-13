# Building CARTS in CLion — CMake profile setup

Notes for opening the CARTS checkout in CLion without the default
`find_package(MLIR)` failure.

## Configure for Build

**Prerequisite:** LLVM and Polygeist must be built first:

```bash
dekk carts build
```

Then, the LLVM and Polygeist should be available in the `build/`.

Paste into **CMake options** (one line, or one `-D` flag per line):

```text
-DMLIR_DIR=<repo>/build/llvm-project/lib/cmake/mlir
-DClang_DIR=<repo>/build/llvm-project/lib/cmake/clang
-DPOLYGEIST_DIR=<repo>/external/Polygeist
-DPOLYGEIST_BUILD_DIR=<repo>/build/polygeist
-DCMAKE_INSTALL_PREFIX=<repo>/.install/carts
-DCMAKE_C_COMPILER=<repo>/.install/llvm/bin/clang
-DCMAKE_CXX_COMPILER=<repo>/.install/llvm/bin/clang++
-DCMAKE_EXE_LINKER_FLAGS=--ld-path=<repo>/.install/llvm/bin/ld64.lld
-DCMAKE_SHARED_LINKER_FLAGS=--ld-path=<repo>/.install/llvm/bin/ld64.lld
-DCMAKE_MODULE_LINKER_FLAGS=--ld-path=<repo>/.install/llvm/bin/ld64.lld
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```



## Debug in CLion

In the Run/Debug config, add Name and Value to the Environment Variables.

```
DYLD_FALLBACK_LIBRARY_PATH=<repo>/.dekk/env/lib
```

And note that `carts-compile` only takes `.mlir` files as input, not the `.c` or `.cpp` files.

Generate the frontend MLIR first, then debug that.
```bash
# from the carts repo root
dekk carts compile samples/dotproduct/dotproduct.c -O3
# writes samples/dotproduct/dotproduct.mlir (or cwd-relative name — check where it lands)
```

Then in CLion, set program arguments to something like:


```bash
/Users/peng599/pppp/carts-project/carts/dotproduct.mlir --O3 --emit-llvm --arts-config /Users/peng599/pppp/carts-project/carts/samples/arts.cfg
```