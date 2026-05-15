# CARTS Developer Guide

## Introduction

This guide provides comprehensive information about writing code that CARTS can compile and execute. CARTS (Compiler and Runtime for Task-based Systems) provides support for C and C++ programs using OpenMP pragmas and specific memory access patterns.

Understanding the supported constructs and memory layouts is essential for writing efficient CARTS applications. This guide covers what works, what doesn't, and how to adapt your code for CARTS.

## Getting Started

### Prerequisites

The following tools must be available on your system:

- **cmake** >= 3.20
- **ninja** (build system)
- **conda**
- **python** >= 3.11
- **git** with submodule support

Run `dekk carts doctor` to verify that all prerequisites are installed and at the correct versions.

### Installation

```bash
git clone <repo-url>
cd carts
python -m pip install --upgrade dekk
dekk carts install
```

This creates or syncs the project-local conda environment, fetches submodules,
builds the full toolchain, installs the bootstrap `clang`/`clang++` pair
required for the pinned LLVM/MLIR build, and makes the `dekk carts ...`
commands
available for the project:

```bash
dekk carts doctor
```

If `dekk carts doctor` reports issues, resolve them before proceeding.

## Language Support

CARTS supports:

- **C code**: Full support for C programs with OpenMP pragmas
- **C++ code**: Full support for C++ programs with OpenMP pragmas

Both languages can use the same OpenMP constructs and memory layout patterns described in this guide.

## OpenMP Pragma Support

CARTS provides comprehensive support for OpenMP pragmas through the staged
compiler pipeline: `ConvertOpenMPToSde` captures source semantics,
`ConvertSdeToCodir` isolates codelet deps/params, and `ConvertCodirToArts`
materializes ARTS DB/EDT objects. The following OpenMP constructs are
supported:

### 1. Parallel Regions

Basic parallel regions are fully supported.

```c
#pragma omp parallel
{
    // Parallel code here
}
```

**Conversion**: Planned in SDE, isolated through CODIR when codelet-shaped, and
materialized as ARTS work/dependency objects by `codir-to-arts`.

### 2. Worksharing Loops

Worksharing loops with various scheduling policies are supported.

```c
#pragma omp parallel for
for (int i = 0; i < N; i++) {
    // Parallel loop body
}
```

**Supported Schedule Clauses**:

- `schedule(static)` - Static work distribution
- `schedule(static, block_size)` - Static with block size
- `schedule(dynamic)` - Dynamic work distribution
- `schedule(dynamic, block_size)` - Dynamic with block size
- `schedule(guided)` - Guided scheduling
- `schedule(auto)` - Automatic scheduling
- `schedule(runtime)` - Runtime-determined scheduling

**Example with scheduling**:

```c
#pragma omp parallel for schedule(static, 4)
for (int i = 0; i < N; i++) {
    process(data[i]);
}
```

### 3. Tasks with Dependencies

Task-based parallelism with data dependencies is a key feature of CARTS.

```c
#pragma omp task depend(in: A) depend(out: B)
{
    compute(A, B);
}
```

**Supported Dependency Types**:

- `depend(in: variable)` - Read dependency
- `depend(out: variable)` - Write dependency
- `depend(inout: variable)` - Read-write dependency

**Multiple Dependencies**:

```c
#pragma omp task depend(in: A[i-1], B[i]) depend(out: C[i])
{
    // Task body
}
```

**Array Element Dependencies**:

```c
for (int i = 0; i < N; i++) {
    #pragma omp task depend(inout: A[i])
    process(&A[i]);
}
```

**Chunk/Section Dependencies**:
CARTS supports array section dependencies for block-based operations:

```c
#pragma omp task depend(in: A[i:BLOCK_SIZE][j:BLOCK_SIZE])
{
    // Process block of A
}
```

### 4. Reductions

Reductions in parallel loops are supported with inline reduction operations.

```c
int sum = 0;
#pragma omp parallel for reduction(+: sum)
for (int i = 0; i < N; i++) {
    sum += array[i];
}
```

### 5. Single and Master Regions

```c
#pragma omp single
{
    // Executed by one thread only
}

#pragma omp master
{
    // Executed by master thread only
}
```

**Conversion**: Both are planned in SDE and must be materialized through the
SDE/CODIR/ARTS boundary. `codir-to-arts` no longer runs a residual SDE-to-ARTS
bridge; surviving SDE operations fail verification.

### 6. Synchronization Constructs

**Barriers**:

```c
#pragma omp barrier
```

**Taskwait**:

```c
#pragma omp taskwait
```

Both are converted to appropriate ARTS synchronization primitives.

### 7. Taskloop

```c
#pragma omp taskloop
for (int i = 0; i < N; i++) {
    // Loop body executed as tasks
}
```

### 8. Atomic Operations

Atomic updates are supported for addition operations:

```c
#pragma omp atomic update
counter += value;
```

**Supported Operations**:

- Integer addition (`arith.addi`)
- Floating-point addition (`arith.addf`)

**Limitation**: Only canonical add operations are currently supported.

### 9. Runtime Functions

The following OpenMP runtime functions are supported:

```c
int tid = omp_get_thread_num();      // Converted to arts.get_current_node
int nthreads = omp_get_num_threads(); // Converted to arts.get_total_nodes
int max_threads = omp_get_max_threads(); // Converted to arts.get_total_nodes
```

## Memory Layout Requirements

CARTS has specific requirements for memory layouts, enforced by the `canonicalize_memrefs` compiler pass. Understanding these requirements is crucial for writing compatible code.

### The Canonicalization Process

The `canonicalize_memrefs` pass performs a three-phase transformation:

1. **Nested Allocation Transformation**: Converts array-of-arrays to multi-dimensional arrays
2. **Array-of-Pointers Transformation**: Eliminates wrapper allocas that store memrefs
3. **OpenMP Dependency Canonicalization**: Transforms dependencies into ARTS format

### Supported Memory Patterns

#### 1. Array-of-Arrays (Recommended for Multi-Dimensional Data)

This is the recommended pattern for multi-dimensional arrays in CARTS:

```c
// Allocate 2D array as array-of-arrays
double **A = malloc(N * sizeof(double*));
for (int i = 0; i < N; i++) {
    A[i] = malloc(M * sizeof(double));
}

// Use in computations
for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
        A[i][j] = compute(i, j);
    }
}

// Clean up
for (int i = 0; i < N; i++) {
    free(A[i]);
}
free(A);
```

**Why this works**: CARTS recognizes this initialization pattern and automatically converts it to a canonical multi-dimensional array representation (`memref<N x M x double>`).

#### 2. Static Multi-Dimensional Arrays

Static arrays with compile-time sizes are naturally supported for basic operations:

```c
#define N 128
#define M 128
float A[N][M];
A[i][j] = value;
```

#### 3. Block/Chunk Access Patterns

For block-based algorithms, CARTS supports chunk dependencies:

```c
#define BLOCK_SIZE 64

double **A = malloc(N * sizeof(double*));
for (int i = 0; i < N; i++) {
    A[i] = malloc(M * sizeof(double));
}

for (int i = 0; i < N; i += BLOCK_SIZE) {
    for (int j = 0; j < M; j += BLOCK_SIZE) {
        #pragma omp task depend(inout: A[i:BLOCK_SIZE][j:BLOCK_SIZE])
        process_block(A, i, j, BLOCK_SIZE);
    }
}
```

### C Code Guidelines for CanonicalizeMemrefs

The `canonicalize_memrefs` pass transforms nested pointer allocations (array-of-arrays) into canonical multi-dimensional arrays. For this transformation to succeed, the C code must follow specific patterns.

#### Initialization Loop Requirements

The pass detects and transforms the following pattern:

```c
// Outer allocation
double **A = malloc(N * sizeof(double*));

// Initialization loop - MUST be simple
for (int i = 0; i < N; i++) {
    A[i] = malloc(M * sizeof(double));
}
```

**Requirements for the init loop**:

1. **No null checks inside the loop** - The loop body should only contain allocations and stores
2. **No early returns or breaks** - The loop must complete all iterations
3. **No results yielded** - The loop should not produce values used by surrounding code

**What works**:

```c
// GOOD: Simple init loop
for (unsigned i = 0; i < N; i++) {
    A[i] = malloc(N * sizeof(double));
    B[i] = malloc(N * sizeof(double));
}
```

**What doesn't work**:

```c
// BAD: Null check inside loop creates complex control flow
for (int i = 0; i < N; i++) {
    A[i] = malloc(N * sizeof(double));
    // This creates scf.for with iter_args
    if (!A[i]) {                        
        fprintf(stderr, "Error\n");
        return 1;
    }
}
```


#### Outer Null Checks Are Fine

Null checks on the outer allocations (before the init loop) are acceptable:

```c
// GOOD: Outer null checks don't affect the init loop transformation
double **A = malloc(N * sizeof(double*));
double **B = malloc(N * sizeof(double*));

if (!A || !B) {
    fprintf(stderr, "Allocation failed\n");
    return 1;
}

// Simple init loop
for (unsigned i = 0; i < N; i++) {
    A[i] = malloc(N * sizeof(double));
    B[i] = malloc(N * sizeof(double));
}
```

#### Dimension Values Must Be Hoistable

The inner dimension sizes must be computable before the outer allocation. The pass will attempt to hoist dimension computations:

```c
// GOOD: Inner size is known before allocation
unsigned N = atoi(argv[1]);
double **A = malloc(N * sizeof(double*));
for (unsigned i = 0; i < N; i++) {
    A[i] = malloc(N * sizeof(double));  // N is already defined
}

// BAD: Inner size computed inside loop (cannot hoist)
for (unsigned i = 0; i < N; i++) {
    int size = compute_size(i);  // Dynamic per-iteration size
    A[i] = malloc(size * sizeof(double));
}
```

#### Summary: Recommended Pattern

```c
int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <N>\n", argv[0]);
        return 1;
    }

    unsigned N = atoi(argv[1]);

    // Outer allocations
    double **A = malloc(N * sizeof(double*));
    double **B = malloc(N * sizeof(double*));

    // Simple init loop - no null checks, no early returns
    for (unsigned i = 0; i < N; i++) {
        A[i] = malloc(N * sizeof(double));
        B[i] = malloc(N * sizeof(double));
    }

    // Use arrays with OpenMP tasks...
    #pragma omp parallel
    #pragma omp single
    {
        for (unsigned i = 0; i < N; i++) {
            #pragma omp task depend(inout: A[i][0])
            process(A[i], N);
        }
    }

    // Cleanup
    for (unsigned i = 0; i < N; i++) {
        free(A[i]);
        free(B[i]);
    }
    free(A);
    free(B);

    return 0;
}
```

### Unsupported Memory Patterns

#### 1. VLA Pointer Casts (NOT Supported)

Variable-length array pointer casts are not supported:

```c
// NOT SUPPORTED
void process(int N, int M, float (*A)[M]) {
    // VLA pointer type
}
```

#### 2. Linearized Multi-Dimensional Access (NOT Supported as-is)

Accessing multi-dimensional data through a linearized pointer with manual index calculation:

```c
// NOT SUPPORTED without explicit rewriting
float *A = malloc(N * M * sizeof(float));
A[i * M + j] = value;  // Manual linearization
```

**Solution**: Rewrite as array-of-arrays (see supported patterns above).

#### 3. Complex Strided Access

Strided access patterns that cannot be canonicalized are not supported. If you encounter errors, consider rewriting as array-of-arrays.

### Verification and Error Messages

If your code uses unsupported memory patterns, CARTS will report a verification error during compilation:

```
Error: memref.load has X indices but the memref has rank Y.
This indicates strided or multi-dimensional access patterns that were not
properly canonicalized. The source code may use VLA pointer casts or other
memory layouts that CARTS does not currently support. Consider rewriting the
code to use explicit multi-dimensional arrays (e.g., malloc for each dimension).
```

**How to fix**: Convert your memory layout to use the array-of-arrays pattern shown above.

## Working Code Examples

This section demonstrates complete patterns that work well with CARTS.

### Example 1: Matrix Multiplication with Block Dependencies

```c
#include <stdlib.h>

#define BLOCK_SIZE 64

void matmul(int N, int M, int K) {
    // Allocate matrices as array-of-arrays
    float **A = malloc(N * sizeof(float*));
    float **B = malloc(K * sizeof(float*));
    float **C = malloc(N * sizeof(float*));

    for (int i = 0; i < N; i++) {
        A[i] = malloc(K * sizeof(float));
    }
    for (int i = 0; i < K; i++) {
        B[i] = malloc(M * sizeof(float));
    }
    for (int i = 0; i < N; i++) {
        C[i] = malloc(M * sizeof(float));
    }

    // Initialize matrices...

    // Task-based matrix multiplication with dependencies
    #pragma omp parallel
    #pragma omp single
    {
        for (int i = 0; i < N; i += BLOCK_SIZE) {
            for (int j = 0; j < M; j += BLOCK_SIZE) {
                for (int k = 0; k < K; k += BLOCK_SIZE) {
                    #pragma omp task \
                        depend(in: A[i:BLOCK_SIZE][k:BLOCK_SIZE], \
                                   B[k:BLOCK_SIZE][j:BLOCK_SIZE]) \
                        depend(inout: C[i:BLOCK_SIZE][j:BLOCK_SIZE])
                    {
                        // Block matrix multiplication
                        multiply_block(A, B, C, i, j, k, BLOCK_SIZE);
                    }
                }
            }
        }
    }

    // Clean up
    for (int i = 0; i < N; i++) free(A[i]);
    for (int i = 0; i < K; i++) free(B[i]);
    for (int i = 0; i < N; i++) free(C[i]);
    free(A); free(B); free(C);
}
```

### Example 2: 1D Stencil with Point Dependencies

```c
void stencil_1d(double *input, double *output, int N) {
    #pragma omp parallel
    #pragma omp single
    {
        for (int i = 1; i < N-1; i++) {
            #pragma omp task \
                depend(in: input[i-1], input[i], input[i+1]) \
                depend(out: output[i])
            {
                output[i] = 0.25 * input[i-1] + 0.5 * input[i] + 0.25 * input[i+1];
            }
        }
    }
}
```

### Example 3: Parallel For with Reduction

```c
double sum_array(double *array, int N) {
    double sum = 0.0;

    #pragma omp parallel for reduction(+: sum) schedule(static, 256)
    for (int i = 0; i < N; i++) {
        sum += array[i];
    }

    return sum;
}
```

### Example 4: Cholesky Factorization with Block Tasks

```c
void cholesky(int N, int BLOCK_SIZE) {
    // Allocate matrix as array-of-arrays
    double **A = malloc(N * sizeof(double*));
    for (int i = 0; i < N; i++) {
        A[i] = malloc(N * sizeof(double));
    }

    int num_blocks = N / BLOCK_SIZE;

    #pragma omp parallel
    #pragma omp single
    {
        for (int k = 0; k < num_blocks; k++) {
            // Diagonal block factorization
            #pragma omp task depend(inout: A[k][k])
            potrf_block(A[k][k], BLOCK_SIZE);

            // Update blocks below diagonal
            for (int i = k+1; i < num_blocks; i++) {
                #pragma omp task \
                    depend(in: A[k][k]) \
                    depend(inout: A[i][k])
                trsm_block(A[k][k], A[i][k], BLOCK_SIZE);
            }

            // Update trailing submatrix
            for (int i = k+1; i < num_blocks; i++) {
                for (int j = k+1; j <= i; j++) {
                    #pragma omp task \
                        depend(in: A[i][k], A[j][k]) \
                        depend(inout: A[i][j])
                    gemm_block(A[i][k], A[j][k], A[i][j], BLOCK_SIZE);
                }
            }
        }
    }

    // Clean up
    for (int i = 0; i < N; i++) free(A[i]);
    free(A);
}
```

## Non-Working Code Examples and Solutions

### Problem 1: Linearized Array Access

**Does NOT work**:

```c
float *matrix = malloc(N * M * sizeof(float));

#pragma omp parallel for
for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
        matrix[i * M + j] = compute(i, j);  // Linearized access
    }
}
```

**Solution - Rewrite as array-of-arrays**:

```c
float **matrix = malloc(N * sizeof(float*));
for (int i = 0; i < N; i++) {
    matrix[i] = malloc(M * sizeof(float));
}

#pragma omp parallel for
for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
        matrix[i][j] = compute(i, j);
    }
}

// Don't forget to free
for (int i = 0; i < N; i++) free(matrix[i]);
free(matrix);
```

### Problem 2: VLA Function Parameters

**Does NOT work**:

```c
void process_matrix(int N, int M, float matrix[][M]) {
    // VLA parameter type
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            matrix[i][j] = 0;
        }
    }
}
```

**Solution - Use array-of-pointers parameter**:

```c
void process_matrix(int N, int M, float **matrix) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            matrix[i][j] = 0;
        }
    }
}

// Caller allocates as array-of-arrays
float **matrix = malloc(N * sizeof(float*));
for (int i = 0; i < N; i++) {
    matrix[i] = malloc(M * sizeof(float));
}
process_matrix(N, M, matrix);
```

### Problem 3: Complex Pointer Arithmetic

**Does NOT work**:

```c
double *data = malloc(N * M * sizeof(double));

for (int i = 0; i < N; i++) {
    double *row = data + i * M;  // Pointer arithmetic
    #pragma omp task depend(inout: row[0:M])
    process_row(row, M);
}
```

**Solution - Use proper 2D allocation**:

```c
double **data = malloc(N * sizeof(double*));
for (int i = 0; i < N; i++) {
    data[i] = malloc(M * sizeof(double));
}

for (int i = 0; i < N; i++) {
    #pragma omp task depend(inout: data[i][0:M])
    process_row(data[i], M);
}

for (int i = 0; i < N; i++) free(data[i]);
free(data);
```

## Reference Examples

CARTS includes an extensive test suite with working examples. These examples demonstrate various parallelization patterns and memory layouts:

**Location**: `samples/`

### Key Examples to Study

1. **`matrixmul/`** - Matrix multiplication with block dependencies
   - Demonstrates 2D array-of-arrays pattern
   - Block-level task dependencies

2. **`array/chunks/` and `rows/chunks/`** - Chunked array/row examples
   - Shows how to use chunk/section syntax: `A[i:SIZE][j:SIZE]`
   - Multi-neighbor dependencies

3. **`cholesky/dynamic/`** - Cholesky factorization
   - Complex multi-block dependencies
   - Dynamic task creation

4. **`stencil/`** - 1D stencil computation
   - Point dependencies
   - Multi-point read dependencies

5. **`parallel_for/`** - Various parallel loop patterns
   - `reduction/` - Reduction operations
   - `chunk/` - Blocked parallel loops with scheduling
   - `block/` - Block-based parallelism

6. **`convolution/`** - 2D convolution
   - C++ example
   - Array-of-arrays in C++

7. **`array/`** - Array element dependencies
   - Single array element per task
   - Sequential dependencies

8. **`rows/deps/`** - Row-wise dependencies
   - Row dependencies in matrix operations

All examples in this directory follow CARTS-compatible patterns and can be used as templates for your own code.

## Common Errors and Solutions

### Error: "memref.load has X indices but the memref has rank Y"

**Cause**: Your code uses memory access patterns that CARTS cannot canonicalize, such as:

- VLA pointer casts
- Complex strided access
- Pointer arithmetic with manual linearization

**Solution**:

1. Identify the problematic array access in your code
2. Rewrite the allocation to use array-of-arrays pattern
3. Update all accesses to use proper multi-dimensional indexing
4. Ensure function parameters accept `type**` instead of VLA types

### Error: Task dependencies not resolved correctly

**Cause**: Dependency syntax may not match expected patterns, or memory layout prevents proper dependency tracking.

**Solution**:

1. Verify array is allocated using array-of-arrays pattern
2. Check dependency syntax matches examples above
3. Ensure all dimension indices and chunks are properly specified

### Build Failures with OpenMP Pragmas

**Cause**: Using OpenMP constructs not yet supported by CARTS.

**Solution**:

1. Check the supported OpenMP constructs list above
2. If using atomic operations, ensure they are canonical add operations
3. For unsupported constructs, consider alternative approaches using supported pragmas

## Best Practices

1. **Always use array-of-arrays for multi-dimensional data**
   - Allocate each dimension explicitly with malloc
   - Initialize in nested loops

2. **Specify dependencies explicitly**
   - Don't rely on implicit task synchronization
   - Use appropriate dependency types (in/out/inout)

3. **Use chunk dependencies for block algorithms**
   - Syntax: `A[i:BLOCK_SIZE][j:BLOCK_SIZE]`
   - Enables fine-grained data dependencies

4. **Test with CARTS examples**
   - Start with examples from `samples/`
   - Modify incrementally to match your needs

5. **Follow C/C++ best practices**
   - Free all allocated memory
   - Check allocation results
   - Avoid memory leaks

## Conclusion

CARTS provides robust support for C and C++ programs using OpenMP pragmas, with specific requirements for memory layouts. By following the patterns described in this guide and studying the provided examples, you can write efficient CARTS applications that leverage task-based parallelism and fine-grained data dependencies.

For more information about CARTS architecture and design, see the other documentation files in the `docs/` directory.
