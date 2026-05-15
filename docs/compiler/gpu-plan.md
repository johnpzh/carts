# GPU Support Plan: NVIDIA + AMD

## Dialect Ownership

GPU work maps onto the four-layer split (`Polygeist -> sde -> codir -> arts -> arts-rt -> LLVM`) as follows:

- **SDE**: GPU eligibility detection, target hint authoring, and MU/CU/SU planning for GPU codelets (trip count, footprint, embarrassingly-parallel proofs, GPU target attrs).
- **CODIR**: GPU codelet isolation (deps/params/body) following the same `IsolatedFromAbove` ABI rules as CPU codelets; thread-indexing ops live in the codelet body.
- **ARTS**: `arts.gpu_edt` and any GPU-specific DB/EDT/epoch objects, including `gpu_launch_config` attributes and DB transfer orchestration.
- **ARTS-RT**: GPU runtime ABI (kernel-launch packing, device-pointer handling, `arts_edt_create_gpu` lowering). The source tree currently exposes these layers under `lib/arts/dialect/core/` (`arts`) and `lib/arts/dialect/rt/` (`arts-rt`) during the in-progress rename.

The pass tables below describe the concrete passes that implement each layer's GPU responsibilities.

## How ARTS GPU Actually Works

Understanding this is the prerequisite for everything else.

ARTS uses the **same-binary function pointer model**:

```
1. User/CARTS writes a __global__ kernel function
2. Everything is compiled into ONE binary by nvcc (or hipcc)
3. arts_edt_create_gpu(myKernel, grid, block, hint) passes the device fn pointer
4. ARTS scheduler calls:
     cudaLaunchKernel((const void*)fn_ptr, grid, block, args, 0, stream)
```

From `fib_gpu.cu` (the canonical ARTS GPU example):

```c
// GPU kernel — compiled into the same binary as host code
__global__ void fib_join(uint32_t paramc, const uint64_t *paramv,
                         uint32_t depc, arts_edt_dep_t depv[]) {
  unsigned int *x   = (unsigned int *)depv[0].ptr;
  unsigned int *res = (unsigned int *)depv[2].ptr;
  (*res) = (*x) + (*y);
}

// Host side — passes function pointer directly
arts_edt_create_gpu(fib_join, 0, NULL, 3, grid, block, &hint);
```

From `gpu_stream_buffer.cu:164` — the actual kernel launch in ARTS:

```c
void *kernel_args[] = {&paramc, &paramv, &depc, &depv};
cudaLaunchKernel((const void *)fn_ptr, grid, block,
                 (void **)kernel_args, 0, arts_gpus[gpu_id].stream);
```

**ARTS does NOT load separate kernel binaries (fatbin/HSACO) at runtime.**
**ARTS does NOT use MLIR's mgpu* abstraction (`mgpuModuleLoad`, `mgpuLaunchKernel`).**
Those belong to a different execution model. ARTS holds a device function pointer and
calls `cudaLaunchKernel` directly.

---

## Architecture: Two Concerns

```
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 1: Kernel Code Generation  (CARTS compiler, MLIR)         │
│                                                                    │
│  CARTS identifies SDE-planned parallel work and outlines its body │
│  as a __global__ function. Uses MLIR gpu:: dialect internally to  │
│  represent thread indexing (gpu::ThreadIdOp, gpu::BlockIdOp).     │
│                                                                    │
│  Output: a __global__ function in the compiled binary +           │
│          a call to arts_edt_create_gpu(outlinedKernel, ...)       │
│                                                                    │
│  For AMD: compile with hipcc instead of nvcc.                     │
│  hipcc accepts __global__, blockIdx, threadIdx identically.       │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  LAYER 2: ARTS EDT Runtime  (external/arts/, compiled by hipcc)  │
│                                                                    │
│  Manages GPU streams, memory, DB transfers, dependency signals.   │
│  Calls cudaLaunchKernel / hipLaunchKernel with the fn pointer.    │
│                                                                    │
│  Currently: CUDA-only  (cudaMalloc, cudaStream_t, etc.)           │
│  For AMD:   Replace with HIP API (hipMalloc, hipStream_t, etc.)   │
│             HIP is the right unification layer here —             │
│             same code path, routes to CUDA on NVIDIA /            │
│             ROCm on AMD at link time.                              │
└──────────────────────────────────────────────────────────────────┘
```

**Where MLIR's NVVM/ROCDL fit**: CARTS uses MLIR gpu:: dialect *internally* for IR
representation and thread indexing ops during GPU materialization. But the final
output is a native `__global__` function compiled by the GPU compiler (nvcc/hipcc),
NOT a separately serialized fatbin/HSACO binary. The `mgpu*` loading infrastructure
is irrelevant to the ARTS model.

---

## Why HIP for Layer 2

The ARTS runtime needs a single code path that runs on both NVIDIA and AMD.
Options:

| | `#ifdef ARTS_GPU_NVIDIA / AMD` | **HIP (chosen)** |
|---|---|---|
| Code paths | 2 (CUDA + ROCm) | 1 |
| NVIDIA support | native CUDA | HIP → CUDA (zero overhead, header-only) |
| AMD support | native ROCm | HIP → ROCm natively |
| `cudaLaunchKernel` → | `cudaLaunchKernel` / `hipLaunchKernel` | `hipLaunchKernel` always |
| Kernel syntax `__global__` | both compilers accept it | hipcc accepts it |
| Build tool | nvcc / hipcc | hipcc only |
| New SDK dependency | none | HIP SDK (on NVIDIA: `libamdhip64` wraps CUDA) |

HIP is the right answer for Layer 2 because ARTS runtime code calls the GPU API at
runtime (streams, mallocs, kernel launches) — exactly what HIP abstracts. The
`#ifdef` approach would duplicate the entire `gpu_runtime.cu`, `gpu_stream.cu`,
`gpu_stream_buffer.cu` logic.

For Layer 1 (kernel generation), `hipcc` compiles `__global__` functions for both
targets — same source, same binary model.

---

## Phase 1: ARTS Runtime → HIP  (`external/arts/`)

### 1.1 New file: `gpu_platform.h`

```c
// external/arts/libs/include/internal/arts/gpu/gpu_platform.h
// Single GPU portability header. HIP routes to CUDA on NVIDIA, ROCm on AMD.

#include <hip/hip_runtime.h>

#define CHECKCORRECT(x)                                                  \
  do {                                                                   \
    hipError_t _err = (x);                                               \
    if (_err != hipSuccess)                                              \
      ARTS_ERROR("GPU error: %s: %s", #x, hipGetErrorString(_err));     \
  } while (0)
```

### 1.2 CUDA → HIP rename (mechanical, all in 5 files)

Replace `#include <cuda_runtime_api.h>` → `#include "arts/gpu/gpu_platform.h"` in:
- `gpu_stream.h`
- `gpu_stream_buffer.h`
- `gpu_lc_sync_functions.cuh`

Full rename table across `gpu_runtime.cu`, `gpu_stream.cu`, `gpu_stream_buffer.cu`,
`gpu_lc_sync_functions.cu`:

| CUDA | HIP |
|---|---|
| `cudaError_t` | `hipError_t` |
| `cudaStream_t` | `hipStream_t` |
| `cudaDeviceProp` | `hipDeviceProp_t` |
| `cudaSuccess` | `hipSuccess` |
| `cudaMalloc` | `hipMalloc` |
| `cudaFree` | `hipFree` |
| `cudaMallocHost` | `hipHostMalloc` |
| `cudaFreeHost` | `hipHostFree` |
| `cudaMemcpy` | `hipMemcpy` |
| `cudaMemcpyAsync` | `hipMemcpyAsync` |
| `cudaMemsetAsync` | `hipMemsetAsync` |
| `cudaMemcpyPeerAsync` | `hipMemcpyPeerAsync` |
| `cudaMemcpyDeviceToHost` | `hipMemcpyDeviceToHost` |
| `cudaMemcpyHostToDevice` | `hipMemcpyHostToDevice` |
| `cudaGetDevice` | `hipGetDevice` |
| `cudaSetDevice` | `hipSetDevice` |
| `cudaGetDeviceCount` | `hipGetDeviceCount` |
| `cudaGetDeviceProperties` | `hipGetDeviceProperties` |
| `cudaMemGetInfo` | `hipMemGetInfo` |
| `cudaStreamCreate` | `hipStreamCreate` |
| `cudaStreamDestroy` | `hipStreamDestroy` |
| `cudaStreamSynchronize` | `hipStreamSynchronize` |
| `cudaStreamQuery` | `hipStreamQuery` |
| `cudaLaunchKernel` | `hipLaunchKernel` |
| `cudaLaunchHostFunc` | `hipLaunchHostFunc` |
| `cudaStreamAddCallback` | `hipStreamAddCallback` |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | `hipOccupancyMaxActiveBlocksPerMultiprocessor` |
| `cudaDeviceCanAccessPeer` | `hipDeviceCanAccessPeer` |
| `cudaDeviceEnablePeerAccess` | `hipDeviceEnablePeerAccess` |
| `cudaDeviceDisablePeerAccess` | `hipDeviceDisablePeerAccess` |
| `cudaGetErrorString` | `hipGetErrorString` |
| `CUDART_VERSION` | `HIP_VERSION` |
| `dim3` | `dim3` (unchanged — HIP defines same type) |
| `__global__`, `blockIdx`, `threadIdx`, `blockDim` | unchanged |

**Extern weak imports** in `gpu_runtime.cu` — update stream type:

```c
// was: cudaStream_t *stream
extern void arts_init_per_gpu(unsigned int node_id, int dev_id,
                               hipStream_t *stream, int argc, char **argv)
    ARTS_WEAK_IMPORT;
extern void arts_fini_per_gpu(unsigned int node_id, int dev_id,
                               hipStream_t *stream) ARTS_WEAK_IMPORT;
```

**Public API** (`external/arts/libs/include/public/arts/gpu.h`): No changes — already CUDA-free.
Add `|| defined(__HIPCC__)` alongside the existing `#ifdef __CUDACC__` guard.

### 1.3 CMake changes (`external/arts/CMakeLists.txt`)

Replace the CUDA detection block with HIP:

```cmake
option(ARTS_USE_GPU "Enable GPU support" ON)

if(ARTS_USE_GPU)
  # hipcc compiles for NVIDIA (via CUDA) and AMD (via ROCm)
  find_program(HIPCC hipcc HINTS /opt/rocm/bin)
  if(HIPCC)
    set(CMAKE_CXX_COMPILER ${HIPCC})
    set(GPU_AVAILABLE ON)
    add_compile_definitions(ARTS_USE_GPU __HIP_PLATFORM_AMD__)
    # On NVIDIA: set __HIP_PLATFORM_NVIDIA__ instead
  else()
    # Fallback to CUDA-only
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
      enable_language(CUDA)
      find_package(CUDAToolkit QUIET)
      set(GPU_AVAILABLE ${CUDAToolkit_FOUND})
    endif()
  endif()
endif()
```

In `gpu/CMakeLists.txt`, set source files to HIP language so CMake routes them to
hipcc:

```cmake
file(GLOB GPU_SOURCES "*.cu")   # hipcc accepts .cu extension
set_source_files_properties(${GPU_SOURCES} PROPERTIES LANGUAGE HIP)
```

---

## Phase 2: Port GPU Compiler Passes to main

**Scope**: `lib/arts/passes/`, `include/arts/`

The GPU pass stack exists on the `gpu` branch. These must land on `main` first.

### Passes to port

| Pass | File | Role |
|---|---|---|
| `GpuEligibilityAnalysis` | `lib/arts/passes/opt/general/GpuEligibilityAnalysis.cpp` | Marks SDE-planned work eligible for GPU offload (trip count + footprint + embarrassingly parallel check) |
| `GpuWorkMaterialization` | `lib/arts/passes/transforms/GpuWorkMaterialization.cpp` | Outlines the planned body to `arts.gpu_edt` with `gpu::ThreadIdOp`/`gpu::BlockIdOp` thread indexing; sets DB modes to `gpu_read`/`gpu_write` |
| `GpuEdtLowering` | `lib/arts/passes/transforms/GpuEdtLowering.cpp` | `arts.gpu_edt` → `arts.edt<task><intranode>` + `gpu_launch_config` attribute |
| `GpuCodegen` | `lib/arts/passes/transforms/GpuCodegen.cpp` | Validation: fails if any `arts.gpu_edt` remains |

Also port from `gpu` branch:

- `ArtsOps.td`: `arts.gpu_edt`, `arts.gpu_memcpy` ops
- `ArtsAttributes.td`: `GpuTargetAttr`, `GpuProfitabilityAttr`, `GpuLaunchConfigAttr`
- `ConvertArtsToLLVM.cpp`: GPU EDT pattern → `artsEdtCreateGpu()` call generation

### What CARTS generates for a GPU loop

Input (user's OpenMP parallel loop):

```c
#pragma omp parallel for
for (int i = 0; i < N; i++)
  c[i] = a[i] + b[i];
```

After CARTS GPU passes, output (conceptually):

```c
// Outlined __global__ kernel (generated by GPU work materialization)
__global__ void _carts_kernel_saxpy(uint32_t paramc, const uint64_t *paramv,
                                    uint32_t depc, arts_edt_dep_t depv[]) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    float *a = (float*)depv[0].ptr;
    float *b = (float*)depv[1].ptr;
    float *c = (float*)depv[2].ptr;
    c[i] = a[i] + b[i];
  }
}

// Host code (generated by ConvertArtsToLLVM)
arts_dim3_t grid  = { (N + 255) / 256, 1, 1 };
arts_dim3_t block = { 256, 1, 1 };
arts_edt_create_gpu(_carts_kernel_saxpy, 0, NULL, 3, grid, block, &hint);
```

ARTS receives the function pointer to `_carts_kernel_saxpy`, schedules it as a GPU
EDT, copies DB data to the GPU, and launches via `hipLaunchKernel(fn_ptr, ...)`.

### Pass pipeline in `Compile.cpp`

```
setupOpenMPToArts()    → GpuEligibilityAnalysis   (--gpu)
setupOpenMPToArts()    → GpuWorkMaterialization    (--gpu)
setupPreLowering()     → GpuEdtLowering + GpuCodegen (--gpu)
setupArtsToLLVM()      → ConvertArtsToLLVM handles gpu_launch_config
```

---

## Phase 3: GPU Target Selection in CARTS

**Scope**: `tools/compile/Compile.cpp`, compiler driver

With Phase 1+2 in place, the only remaining compiler-side change is selecting the
right compiler toolchain and architecture:

### New CLI flags

```cpp
static cl::opt<bool> EnableGpu("gpu",
    cl::desc("Enable GPU offloading"));

static cl::opt<std::string> GpuArch("gpu-arch",
    cl::desc("GPU architecture: sm_80 (NVIDIA) or gfx90a (AMD)"),
    cl::init(""));
```

The `--gpu-vendor` distinction is handled automatically:
- NVIDIA: `hipcc --offload-arch=sm_80` (hipcc wraps nvcc for NVIDIA targets)
- AMD:    `hipcc --offload-arch=gfx90a`

Both produce a single binary with `__global__` kernels callable via `hipLaunchKernel`.

MLIR's `gpu::ThreadIdOp`, `gpu::BlockIdOp`, `gpu::BlockDimOp` ops generated by
GPU materialization lowers to the appropriate `threadIdx.x`, `blockIdx.x` etc. via the
standard SCF/Affine → LLVM lowering, then `hipcc` compiles the result.

---

## File Change Summary

### Phase 1 — ARTS runtime (external/arts/)

```
NEW   external/arts/libs/include/internal/arts/gpu/gpu_platform.h
MOD   external/arts/libs/include/internal/arts/gpu/gpu_stream.h
MOD   external/arts/libs/include/internal/arts/gpu/gpu_stream_buffer.h
MOD   external/arts/libs/include/internal/arts/gpu/gpu_lc_sync_functions.cuh
MOD   external/arts/libs/include/public/arts/gpu.h       (__HIPCC__ guard)
MOD   external/arts/libs/src/core/gpu/gpu_runtime.cu
MOD   external/arts/libs/src/core/gpu/gpu_stream.cu
MOD   external/arts/libs/src/core/gpu/gpu_stream_buffer.cu
MOD   external/arts/libs/src/core/gpu/gpu_lc_sync_functions.cu
MOD   external/arts/CMakeLists.txt                        (HIP detection)
MOD   external/arts/libs/src/core/gpu/CMakeLists.txt      (LANGUAGE HIP)
MOD   Makefile                                            (-DARTS_USE_GPU=ON)
```

### Phase 2 — GPU compiler passes (port from gpu branch)

```
NEW   lib/arts/passes/Analysis/GpuEligibilityAnalysis.cpp
NEW   lib/arts/passes/transforms/GpuWorkMaterialization.cpp
NEW   lib/arts/passes/transforms/GpuEdtLowering.cpp
NEW   lib/arts/passes/transforms/GpuCodegen.cpp
MOD   include/arts/passes/Passes.td
MOD   include/arts/passes/Passes.h
MOD   include/arts/ArtsOps.td                    (arts.gpu_edt, arts.gpu_memcpy)
MOD   lib/arts/passes/transforms/ConvertArtsToLLVM.cpp
MOD   tools/compile/Compile.cpp                  (GPU pass pipeline)
```

### Phase 3 — Compiler driver

```
MOD   tools/compile/Compile.cpp      (--gpu, --gpu-arch flags)
MOD   tools/carts_cli.py             (pass --gpu-arch to compiler)
```

---

## Constraints

- Phase 2 blocks Phase 3. GPU passes must land on `main` before any codegen work.
- Phase 1 requires HIP SDK: on NVIDIA machines, `libamdhip64` wraps CUDA (header-only
  on NVIDIA, no ROCm install needed); on AMD machines, ROCm provides HIP natively.
- `__global__`, `blockIdx`, `threadIdx`, `blockDim` are identical in CUDA and HIP.
  User-written GPU kernels need no changes.
- MLIR's `MLIRGPUToNVVMTransforms` / `MLIRGPUToROCDLTransforms` / `mgpu*` are NOT
  used in the ARTS execution model. ARTS uses `hipLaunchKernel` with function
  pointers, not separately-loaded kernel binaries.
