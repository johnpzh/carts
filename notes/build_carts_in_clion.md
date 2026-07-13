# Building CARTS in CLion — CMake profile setup

Notes for opening the CARTS checkout in CLion without the default
`find_package(MLIR)` failure.

## TL;DR

| Symptom | Root cause | Fix |
|---------|-----------|-----|
| `Could not find MLIRConfig.cmake` / `MLIR_DIR-NOTFOUND` | CLion runs bare CMake with no `MLIR_DIR`; the LLVM **source** tree does not contain package config files | Point `MLIR_DIR` at the **built** LLVM tree under `<repo>/build/llvm-project/lib/cmake/mlir` |
| Configure succeeds but link fails with wrong toolchain | Wrong compiler or linker | Use `.install/llvm/bin/clang++` and `ld64.lld` from the same install prefix |
| `undefined symbol: DbReleaseOp::build(..., DbReleaseType)` | ODS declares a custom builder that had no C++ definition; Debug keeps the generated `create` that references it | Implement the builder in `lib/carts/dialect/arts/IR/Dialect.cpp` (fixed in-tree) |
| POST_BUILD tries to write `/usr/local/bin/carts-compile` | Default `CMAKE_INSTALL_PREFIX` is `/usr/local` while tests sync the binary there | Set `-DCMAKE_INSTALL_PREFIX=<repo>/.install/carts` (or disable the sync; see below) |
| Just need navigation/indexing | Full CLion CMake configure is optional | Load `<repo>/build/carts/compile_commands.json` and keep building with `dekk carts build` |

**Prerequisite:** LLVM and Polygeist must be built first:

```bash
dekk carts build --llvm
dekk carts build --polygeist
dekk carts build
```

---

## Path model

CARTS does not configure against the LLVM source checkout directly. CMake
needs the **build-tree** package configs produced when LLVM is built.

| Path | Role |
|------|------|
| `external/Polygeist/llvm-project/` | LLVM/MLIR **source** (no `MLIRConfig.cmake` here) |
| `build/llvm-project/lib/cmake/mlir/` | `MLIRConfig.cmake` — set `MLIR_DIR` here |
| `build/llvm-project/lib/cmake/clang/` | `ClangConfig.cmake` — set `Clang_DIR` here |
| `.install/llvm/bin/clang++` | Compiler the project Makefile uses |
| `.install/llvm/bin/ld64.lld` | Linker on macOS (`ld.lld` on Linux) |
| `build/polygeist/` | Built Polygeist libraries (required by CARTS `CMakeLists.txt`) |
| `build/carts/` | Official dekk-managed CARTS build tree |

Generated artifacts live under `CARTS_HOME` when set, otherwise under the
checkout root. If you use a custom `carts.config` or `CARTS_HOME`, replace
`<repo>` below with that root.

---

## Option A — Compilation database only (recommended)

Use this when CLion is mainly for editing, go-to-definition, and refactoring.
Keep building and testing through dekk.

1. Build once: `dekk carts build`
2. In CLion: **Settings → Build, Execution, Deployment → Compilation Database**
3. Set the path to:

   ```
   <repo>/build/carts/compile_commands.json
   ```

4. Reload the project.

CLion will index using the same flags and include paths as the official build.
No separate `cmake-build-debug` profile is required.

---

## Option B — Full CLion CMake profile

Use this when you want CLion to configure and build CARTS itself (e.g. a
`cmake-build-debug` directory at the repo root).

### 1. Open the project

Open the CARTS checkout root (the directory containing the top-level
`CMakeLists.txt`).

### 2. Create or edit a CMake profile

**Settings → Build, Execution, Deployment → CMake**

Suggested profile settings:

| Field | Value |
|-------|-------|
| **Build type** | `Debug` (IDE-friendly) or `Release` (matches `dekk carts build`) |
| **Build directory** | `<repo>/cmake-build-debug` (or any directory you prefer) |
| **Generator** | Ninja |

### 3. CMake options

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

On Linux, replace `ld64.lld` with `ld.lld`.

Optional for IDE-only builds (skip syncing `carts-compile` into the install
prefix after every link):

```text
-DCARTS_TESTS_REQUIRE_INSTALLED_RUNNER=OFF
```

Example with an absolute checkout path:

```text
-DMLIR_DIR=/Users/you/carts/build/llvm-project/lib/cmake/mlir
-DClang_DIR=/Users/you/carts/build/llvm-project/lib/cmake/clang
-DPOLYGEIST_DIR=/Users/you/carts/external/Polygeist
-DPOLYGEIST_BUILD_DIR=/Users/you/carts/build/polygeist
-DCMAKE_INSTALL_PREFIX=/Users/you/carts/.install/carts
-DCMAKE_C_COMPILER=/Users/you/carts/.install/llvm/bin/clang
-DCMAKE_CXX_COMPILER=/Users/you/carts/.install/llvm/bin/clang++
-DCMAKE_EXE_LINKER_FLAGS=--ld-path=/Users/you/carts/.install/llvm/bin/ld64.lld
-DCMAKE_SHARED_LINKER_FLAGS=--ld-path=/Users/you/carts/.install/llvm/bin/ld64.lld
-DCMAKE_MODULE_LINKER_FLAGS=--ld-path=/Users/you/carts/.install/llvm/bin/ld64.lld
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

These mirror the variables passed by the project `Makefile` `build` target,
plus an explicit install prefix so CLion does not default to `/usr/local`.

### 4. Reload CMake

Click **Reload CMake Project** (or restart CLion). Configure should print:

```
Searching for MLIRConfig.cmake in: <repo>/build/llvm-project/lib/cmake/mlir
Using MLIRConfig.cmake in: ...
```

### 5. Build from CLion

Use the CLion build/run targets, or continue using `dekk carts build` for
compiler/runtime work that matches CI and local test flows.

---

## Troubleshooting

### `MLIR_DIR-NOTFOUND`

LLVM has not been built, or `MLIR_DIR` points at the source tree instead of
the build tree. Confirm this file exists:

```
<repo>/build/llvm-project/lib/cmake/mlir/MLIRConfig.cmake
```

If missing: `dekk carts build --llvm`.

### Polygeist library not found

Confirm `build/polygeist/lib/` exists. If missing: `dekk carts build --polygeist`.

### Stale CMake cache after branch switch or rebuild

Delete the CLion build directory (e.g. `cmake-build-debug/`) and reload CMake,
or run `dekk carts build --clean` for the managed trees under `build/`.

### Debug vs Release mismatch

`dekk carts build` uses **Release**. A CLion **Debug** profile is fine for
IDE iteration but produces different binaries and optimization behavior than
the dekk-managed build in `build/carts/`.

### `undefined symbol: DbReleaseOp::build(..., DbReleaseType)`

`ArtsOps.td` declares two custom builders for `arts.db_release`:

```tablegen
let builders = [
  OpBuilder<(ins "Value":$source)>,
  OpBuilder<(ins "Value":$source, "DbReleaseType":$release_type)>
];
```

TableGen emits `create(..., DbReleaseType)` which calls
`DbReleaseOp::build(..., Value, DbReleaseType)`. That `build` must be
implemented in C++ (`lib/carts/dialect/arts/IR/Dialect.cpp`). The
Value-only overload existed; the enum overload was missing.

**Why CLion/Debug fails while `dekk carts build` may appear fine:** Debug
links keep the unused generated `create` and surface the missing symbol.
Release often dead-strips it, so the official Release build can succeed
without the definition.

**Fix:** rebuild after the in-tree `Dialect.cpp` implementation is present.
Reload/rebuild the CLion target; no extra CMake flags are required for this
error.

### POST_BUILD copies into `/usr/local/bin`

With tests enabled, `tools/compile/CMakeLists.txt` syncs `carts-compile` into
`${CMAKE_INSTALL_PREFIX}/bin` after linking. CLion’s default install prefix
is `/usr/local`, so the link step may try to write `/usr/local/bin`.

Set `-DCMAKE_INSTALL_PREFIX=<repo>/.install/carts`, or
`-DCARTS_TESTS_REQUIRE_INSTALLED_RUNNER=OFF` for IDE builds that do not need
the installed runner.

---

## Reference — what the Makefile passes

The official CARTS configure step (from `Makefile`) sets:

```make
-DMLIR_DIR=$(LLVM_BUILD_DIR)/lib/cmake/mlir
-DClang_DIR=$(LLVM_BUILD_DIR)/lib/cmake/clang
-DPOLYGEIST_BUILD_DIR=$(POLYGEIST_BUILD_DIR)
-DPOLYGEIST_DIR=$(POLYGEIST_DIR)
-DCMAKE_C_COMPILER=$(LLVM_INSTALL_DIR)/bin/clang
-DCMAKE_CXX_COMPILER=$(LLVM_INSTALL_DIR)/bin/clang++
```

with `LLVM_BUILD_DIR=<repo>/build/llvm-project` and
`LLVM_INSTALL_DIR=<repo>/.install/llvm` by default.
