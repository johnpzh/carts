# Building & testing CARTS on macOS — findings

Notes from debugging a fully-red `dekk carts test` on macOS (Apple Silicon,
Python 3.13, conda clang). Three independent problems were stacked; the first
two broke the whole suite, the third is a pre-existing unrelated fixture bug.

## TL;DR

| # | Symptom | Root cause | Fix |
|---|---------|-----------|-----|
| 1 | `ModuleNotFoundError: No module named 'scripts'` | `lit.cfg.py` front-loaded `tools/scripts` on `sys.path`; `tools/scripts/platform.py` shadowed stdlib `platform` in lit's spawn workers | `append` instead of `insert(0, ...)` in `lit.cfg.py` |
| 2 | `dyld: Symbol not found: __ZnamSt19__type_descriptor_t` on every tool | managed libc++abi lacks C++26 type-aware `operator new/new[]`; `DYLD_LIBRARY_PATH` made it shadow the complete system libc++abi | use `DYLD_FALLBACK_LIBRARY_PATH` on macOS (`tools/scripts/local_config.py`) |
| 3 | `arts-rt/.../no_core_cps_schema_after_pre_lowering.mlir` fails | test consumes gitignored/generated `samples/parallel_for/loops/parallel_for.mlir`, not present in checkout | (open) regenerate sample or fix the test |

Result: `dekk carts test` went from **0/263** to **262/263** (only #3 remains).

---

## Problem 1 — stdlib `platform` shadowed in lit spawn workers

`dekk carts test` spewed, once per worker:

```
ModuleNotFoundError: No module named 'scripts'
  File ".../.install/llvm/bin/llvm-lit", line 22, in <module>
    import platform
  File ".../tools/scripts/platform.py", line 28, in <module>
    from scripts.local_config import (...)
```

**Mechanism**

- `lit.cfg.py` did `sys.path.insert(0, tools/scripts)` so it could
  `from local_config import ...`. That directory also contains `platform.py`,
  which shadows Python's standard-library `platform`.
- macOS + Python 3.13 run lit's parallel workers with the **spawn** start
  method. Each worker restores the parent's `sys.path` and re-execs `llvm-lit`,
  whose top-level `import platform` (before `lit.cfg.py` ever runs) then
  resolves to our module. Its `from scripts.local_config import ...` fails
  because only `tools/scripts` (not `tools/`) is on the path.
- Only breaks the parallel worker pool; a single-file `dekk carts lit` (no pool)
  still passed, which is why it looked intermittent.

**Fix** — `lit.cfg.py`: append the dir instead of front-loading it, so the
standard library wins for `import platform` while `local_config` still resolves.

```python
if tools_scripts_dir not in sys.path:
    sys.path.append(tools_scripts_dir)   # NOT insert(0, ...)
```

Repro of the shadow:
```
python3 -c "import sys; sys.path.insert(0,'tools/scripts'); import platform"   # -> ModuleNotFoundError
python3 -c "import sys; sys.path.append('tools/scripts');  import platform"    # -> stdlib, fine
```

---

## Problem 2 — type-aware `operator new` / libc++abi ABI skew

After fixing #1, every test failed at a native dyld error:

```
dyld: Symbol not found: __ZnamSt19__type_descriptor_t
      = operator new[](unsigned long, std::__type_descriptor_t)   # C++26 type-aware allocation
  Referenced from: .../carts-compile  (and FileCheck, llvm-ar, clang, opt, ...)
  Expected as weak-def export from some loaded dylib
```

**Mechanism (this is the important one)**

- These operators are **C++26 type-aware allocation** and live in **libc++abi**,
  not libc++.
- LLVM is bootstrapped with **conda clang** (`.dekk/env/bin/clang++`), whose
  libc++ headers target the type-aware ABI. Real code (e.g.
  `std::vector<std::string>`) makes the compiler emit **weak** references to
  those operators. (A trivial `new int[4]` does not — which is why small repros
  looked clean.)
- macOS's **system** `/usr/lib/libc++abi.dylib` provides these operators (it is
  in the dyld shared cache, so `nm` can't see them — confirm with
  `DYLD_PRINT_BINDINGS=1`, which shows `using ... in /usr/lib/libc++abi.dylib`).
- The project's **managed** `.install/llvm/lib/libc++abi` (built from the older
  in-tree libcxxabi) exports **none** of them.
- lit/dekk force the managed lib dir onto `DYLD_LIBRARY_PATH`, which dyld
  searches **before** each binary's install-names/rpaths. So the incomplete
  managed libc++abi shadows the complete system one → the weak-def can't be
  found → every tool aborts. Even `llvm-ar` aborts during builds.
- Run **bare**, the tools work (system libc++abi resolves the operators); it is
  only the forced `DYLD_LIBRARY_PATH` that breaks them.

**Why a rebuild does not help** — a clean `--llvm` rebuild uses the same conda
clang (still emits the refs) and rebuilds the same managed libc++abi (still
missing the operators). Confirmed: after `dekk carts build --llvm --clean`, the
freshly-built `llvm-ar` was still broken and the fresh libc++abi still exported 0
type-aware operators.

**Fix** — `tools/scripts/local_config.py` `runtime_library_env_vars()`: on macOS
return `DYLD_FALLBACK_LIBRARY_PATH` instead of `DYLD_LIBRARY_PATH`.

```
DYLD_LIBRARY_PATH           broken: managed libc++abi shadows system  -> Symbol not found
DYLD_FALLBACK_LIBRARY_PATH  works:  system libc++abi resolves; managed libs via rpath
```

The fallback variable is searched **after** install-names/rpaths, so managed
libc++/libomp/libarts still resolve through each binary's rpath
(`@loader_path/../lib`), while the system libc++abi supplies the type-aware
operators. Verified: `FileCheck`/`carts-compile` run under the managed path and
the suite goes green.

Do **not** revert to `DYLD_LIBRARY_PATH` without first making the managed
libc++abi export the type-aware operators.

**Deeper (unfixed) cause** — bootstrap-compiler skew: the conda clang is newer
than the in-tree libcxxabi. A structural fix would bootstrap LLVM with a matched
clang, or bump the in-tree libcxxabi to one that defines these operators.

Handy commands used:
```
# reproduce / confirm the abort
DYLD_LIBRARY_PATH=.install/llvm/lib .install/llvm/bin/FileCheck --version      # aborts
DYLD_FALLBACK_LIBRARY_PATH=.install/llvm/lib .install/llvm/bin/FileCheck --version  # works

# see who actually provides the symbol
DYLD_PRINT_BINDINGS=1 .dekk/env/bin/clang++-built-binary 2>&1 | grep type_descriptor

# check a libc++abi's type-aware exports
nm -gU .install/llvm/lib/libc++abi.1.0.dylib | grep 'Zn[aw]mSt19__type_descriptor_t'
```

---

## Problem 3 — stale/generated sample fixture (still open, unrelated)

`lib/carts/dialect/arts-rt/test/lowering/no_core_cps_schema_after_pre_lowering.mlir`
has a second RUN line that compiles
`%samples_dir/parallel_for/loops/parallel_for.mlir`, which does not exist:

```
[ERROR] [compile] Could not open input file: .../samples/parallel_for/loops/parallel_for.mlir
```

`parallel_for.mlir` is **gitignored** (a generated artifact) and was never
committed; only `parallel_for.c` (+ a built `parallel_for`) exist in that dir.
Independent of problems 1 and 2. Options: regenerate the sample `.mlir` as a
test prerequisite, or update/remove that RUN line.

---

## Files changed

- `lit.cfg.py` — `append` (not `insert(0)`) for `tools/scripts` on `sys.path`.
- `tools/scripts/local_config.py` — `runtime_library_env_vars()` returns
  `DYLD_FALLBACK_LIBRARY_PATH` on macOS.

Both are macOS-specific; Linux (`LD_LIBRARY_PATH`) behavior is unchanged.
