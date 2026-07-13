# Build CARTS on macOS

Reference:

1. [https://github.com/randreshg/carts/blob/main/docs/getting-started.md](https://github.com/randreshg/carts/blob/main/docs/getting-started.md)

Run

```bash
dekk carts install
```

When failed for `libclang-cpp22.1-22.` and in the `install.log`, it says

```bash
...
-- GPU support disabled (ARTS_USE_GPU=OFF)
-- Will build only standard ARTS library (GPU support disabled)
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
-- Found Threads: TRUE
CMake Error at CMakeLists.txt:287 (message):
  ARTS_USE_RDMA=ON requires librdmacm/RSockets, but pkg-config could not find
  librdmacm

-- Configuring incomplete, errors occurred!
ninja: Entering directory `/Users/peng599/pppp/carts-project/carts/build/arts'
ninja: error: loading 'build.ninja': No such file or directory
make: *** [Makefile:214: arts] Error 1
  ✗ Building ARTS runtime... failed
```

It’s because `dekk carts build --arts` turns on **RDMA/RoCE** by default (`ARTS_USE_RDMA=ON`). ARTS then requires `librdmacm` (via `pkg-config`). That library is part of the Linux RDMA stack and is **not available on macOS**, so CMake fails before `build.ninja` is generated.

To workaround it, run

```bash
export SDKROOT="$(xcrun --show-sdk-path)" # for link and hwloc configure succeed
dekk carts build --arts --no-rdma
```

On macOS, `<ucontext.h>` might cause errors unless `_XOPEN_SOURCE` is defined.

In the `external/arts/libs/src/core/system/debug.c`, add `_XOPEN_SOURCE` and `_DARWIN_C_SOURCE` before including `<ucontext.h>`.

```cpp
#define _XOPEN_SOURCE 600
#define _DARWIN_C_SOURCE
#include <ucontext.h>
```

Then do

```bash
dekk carts build --arts --no-rdma --clean
```

Then continue to build

```bash
dekk carts build

```

To run test `dekk carts test`, in the file `lit.cfg.py`, change

```python
sys.path.insert(0, tools_scripts_dir)

```

to

```python
sys.path.append(tools_scripts_dir)
```

In the file `tools/scripts/local_config.py`, change

```python
    if is_macos_platform(info):
        return ("DYLD_LIBRARY_PATH",)
    if is_linux_platform(info):
        return ("LD_LIBRARY_PATH",)
    return ("DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH")
```

to

```python
    if is_macos_platform(info):
        return ("DYLD_FALLBACK_LIBRARY_PATH",)
    if is_linux_platform(info):
        return ("LD_LIBRARY_PATH",)
    return ("DYLD_FALLBACK_LIBRARY_PATH", "LD_LIBRARY_PATH")
```

Then run

```bash
dekk carts build --llvm --clean && dekk carts build --clean
```

When compiling source code, please specify the `arts-config` file

```bash
dekk carts compile samples/dotproduct/dotproduct.c -O3 -o dotproduct --arts-config samples/arts.cfg 
```

